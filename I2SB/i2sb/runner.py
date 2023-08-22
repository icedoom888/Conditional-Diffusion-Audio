# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

import distributed_util as dist_util
from . import util
from .diffusion import Diffusion

from ipdb import set_trace as debug

from torchaudio import save as save_audio
from audio_diffusion_pytorch import UNetV0
from omegaconf import OmegaConf
import sys
import os
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_file_path, "..", "..", "vits"))
sys.path.append(os.path.join(this_file_path, "..", ".."))
from vits.utils_diffusion import load_vits_model, get_Z_to_audio
from custom_dataset import *

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        #noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

        # load the model args
        self.conf = OmegaConf.load(opt.model_conf_path)
        model_args = self.conf.model

        #self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1) #TODO use own network here and change the ways we call the network
        self.net = UNetV0(
            dim=2,
            in_channels=2 if not model_args.use_initial_image else 3, #IMAGE | MASK | OPTIONAL(INIT IMAGE)
            out_channels=1, # 1 for the output image
            channels=list(model_args.channels), # U-Net: number of channels per layer
            factors=list(model_args.factors), # U-Net: image size reduction per layer
            items=list(model_args.layers), # U-Net: number of layers
            attentions=list(model_args.attentions), # U-Net: number of attention layers
            cross_attentions=list(model_args.cross_attentions), # U-Net: number of cross attention layers
            attention_heads=model_args.attention_heads, # U-Net: number of attention heads per attention item
            attention_features=model_args.attention_features , # U-Net: number of attention features per attention item
            use_text_conditioning=False, # U-Net: enables text conditioning (default T5-base)
            use_additional_time_conditioning=model_args.use_additional_time_conditioning, # U-Net: enables additional time conditionings
            use_embedding_cfg=model_args.use_embedding_cfg, # U-Net: enables classifier free guidance
            embedding_max_length=model_args.embedding_max_length, # U-Net: text embedding maximum length (default for T5-base)
            embedding_features=model_args.embedding_features, # text embedding dimensions, is used for CFG
            resnet_groups = 8,
            use_modulation = True, # needed for timesteps
            modulation_features = 1024,
            use_time_conditioning = True,
        )

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load: # TODO check if this works
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, data=None):
        data = next(loader) if data is None else data
        
        clean_img = data["z_audio"]
        x0_mask = data["z_audio_mask"]
        corrupt_img = data["z_text"]
        x1_mask = data["z_text_mask"]
        embeds = data["clap_embed"]
        
        if opt.conf_file["training"]["dataset"] == "VCTK":
            sid = data["sid"]
            Z_MEAN = VCTK_MEAN_TEXT
            Z_STD = VCTK_STD_TEXT
        else:
            sid = None
            Z_MEAN = LJS_MEAN_TEXT
            Z_STD = LJS_STD_TEXT

        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        embeds = embeds.detach().to(opt.device)
        x0_mask = x0_mask.detach().to(opt.device)
        x1_mask = x1_mask.detach().to(opt.device)

        cond = x1.detach() if opt.cond_x1 else None # TODO load the conditioning here

        if opt.add_x1_noise:
            x1 = x1 + torch.normal(mean=Z_MEAN, std=Z_STD, size=x1.shape, device=x1.device)

        assert x0.shape == x1.shape

        return x0, x1, sid, cond, embeds, x0_mask, x1_mask, data

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device], find_unused_parameters=False)
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        net.train()

        # load vits
        if opt.global_rank == 0:
            if opt.conf_file["training"]["dataset"] == "LJS":
                conf_path = "ljs_base.json"
                ckpt_path = "pretrained_ljs.pth"
            else:
                conf_path = "vctk_base.json"
                ckpt_path = "pretrained_vctk.pth"

            self.vits_model, hps = load_vits_model(
                hps_path=os.path.join(opt.conf_file["training"]["vits_root"], "configs", conf_path),
                checkpoint_path=os.path.join(opt.conf_file["training"]["vits_root"], ckpt_path)
                )
            self.z_to_audio = get_Z_to_audio(self.vits_model)

        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                # TODO this is actually okay, the x0 is the end, x1 will be the start, mask is None, is not used and cond will be x1
                x0, x1, sid, cond, embeds, x0_mask, x1_mask, y_mask, offset, z_length = self.sample_batch(opt, train_loader)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode) # gets the noisy latent at timestep t
                label = self.compute_label(step, x0, xt)

                # concat with mask for guidance in varying length inputs
                if self.conf.model.use_data_mask:
                    xt = torch.cat([xt, x1_mask], dim=1)

                # concat x1 and cond for image/x1 conditioning
                if self.conf.model.use_initial_image:
                    xt = torch.cat([xt, x1], dim=1)

                pred = net(
                    xt, # latent sample at timestep t
                    time = step, # timestep 
                    features = x1.mean(-3) if self.conf.model.use_additional_time_conditioning else None, # embeds an image to additional embeddings that are added to the time embeddings TODO mload configuration of the embedder from the config file
                    embedding = embeds, # embedding for CFG
                    embedding_mask_proba = 0.1
                )

                assert label.shape == pred.shape
                
                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())
            
            if it % 1000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    if it % 5000 == 0:
                        torch.save({
                            "net": self.net.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "sched": sched.state_dict() if sched is not None else sched,
                        }, opt.ckpt_path / f"{it}.pt")
                    
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % 500 == 0: # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, x1_mask=None, embeds=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True, cfg=1.0):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, nfe={nfe}, log_steps={log_steps}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step, cfg=1.0):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                
                xt_total = None

                # concat with mask for guidance in varying length inputs
                if self.conf.model.use_data_mask:
                    assert x1_mask is not None, "x1_mask is None, but use_data_mask is True"
                    xt_total = torch.cat([xt, x1_mask], dim=1)

                # concat x1 and cond for image/x1 conditioning
                if self.conf.model.use_initial_image:
                    xt_total = torch.cat([xt_total if xt_total is not None else xt, x1], dim=1)

                out = self.net(
                    xt if xt_total is None else xt_total, # latent sample at timestep t
                    time = step, # timestep 
                    features = x1.mean(-3) if self.conf.model.use_additional_time_conditioning else None, # embeds an image to additional embeddings that are added to the time embeddings
                    embedding = embeds, # embedding for CFG
                    embedding_scale = cfg
                )

                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose, cfg=cfg
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        x0, x1, sid, cond, embeds, x0_mask, x1_mask, offset, z_length = self.sample_batch(opt, val_loader)
        sid = sid.squeeze() if sid is not None else None

        x1 = x1.to(opt.device) # TODO load the actual target image

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, x1_mask=x1_mask, embeds=embeds, cond=cond, nfe=200, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        batch, len_t, *xdim = xs.shape

        log.info("Collecting tensors ...")
        img_target   = all_cat_cpu(opt, log, x0)
        img_source = all_cat_cpu(opt, log, x1)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        assert img_target.shape == img_source.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid(img, nrow=nrow))

        img_target_pred = xs[:, 0, ...]

        y_masks_audio = data["y_mask_audio"]
        y_masks_text = data["y_mask_text"]

        # logging audio
        for i in range(batch):
            sample = img_target_pred[i]
            gt = x0[i]
            start = x1[i]
            mask = y_mask[i]
            sid_i = torch.LongTensor([int(sid[i])]).cuda() if sid is not None else None

            sample = sample[:, :, offset[i]:offset[i]+z_length[i]]
            gt = gt[:, :, offset[i]:offset[i]+z_length[i]]
            mask = mask[:, offset[i]:offset[i]+z_length[i]]

            # pass through vocoder
            model_audio =  self.z_to_audio(z=sample.cuda(), y_mask=mask.cuda(), sid=sid_i).cpu().squeeze(0)
            gt_audio = self.z_to_audio(z=gt.cuda(), y_mask=mask.cuda(), sid=sid_i).cpu().squeeze(0)
            start_audio = self.z_to_audio(z=start.cuda(), y_mask=mask.cuda(), sid=sid_i).cpu().squeeze(0)

            # save
            sample_path = os.path.join(opt.ckpt_path, "samples", str(it), f"model_audio_{i}.wav")
            gt_path = os.path.join(opt.ckpt_path, "samples", str(it), f"gt_audio_{i}.wav")
            start_audio_path = os.path.join(opt.ckpt_path, "samples", str(it), f"start_audio_{i}.wav")

            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            os.makedirs(os.path.dirname(gt_path), exist_ok=True)
            os.makedirs(os.path.dirname(start_audio_path), exist_ok=True)

            save_audio(sample_path, model_audio, 22050)
            save_audio(gt_path, gt_audio, 22050)
            save_audio(start_audio_path, start_audio, 22050)

            self.writer.add_sound(step=it, caption=f"model_audio_{i}", key="model_audio", sound_path=sample_path)
            self.writer.add_sound(step=it, caption=f"gt_audio_{i}", key="gt_audio", sound_path=gt_path)
            self.writer.add_sound(step=it, caption=f"start_audio_{i}", key="start_audio", sound_path=start_audio_path)

        log.info("Logging images ...")
        log_image("image/clean",   scale_0_1(img_target)) # target image
        log_image("image/corrupt", scale_0_1(img_source)) # source image
        log_image("image/recon",   scale_0_1(img_target_pred))
        log_image("debug/pred_clean_traj", scale_0_1(pred_x0s.reshape(-1, *xdim)), nrow=len_t)
        log_image("debug/recon_traj",      scale_0_1(xs.reshape(-1, *xdim)),      nrow=len_t)

        #log.info("Logging accuracies ...")
        #log_accuracy("accuracy/clean",   img_clean)
        #log_accuracy("accuracy/corrupt", img_corrupt)
        #log_accuracy("accuracy/recon",   img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
