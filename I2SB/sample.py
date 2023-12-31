# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug
from custom_dataset import LJS_Latent, LJSSlidingWindow, VCTKVitsLatents
from torchaudio import save as save_audio
from einops import rearrange


import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_file_path, "..", "..", "vits"))
sys.path.append(os.path.join(this_file_path, "..", ".."))
from vits.utils_diffusion import load_vits_model, get_Z_to_audio, get_Z_preflow_to_audio, get_text_embedder
import yaml
import time
from train import RESULT_DIR

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx}, {end_idx}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    else:
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        x1 = corrupt_img.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, x1, mask, cond, y

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build corruption method
    #corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # build imagenet val dataset
    if opt.conf_file.training.dataset == "LJS":
        val_dataset = LJSSlidingWindow(root=opt.conf_file.training.data_root, mode="val", normalize=False)
    elif opt.conf_file.training.dataset == "VCTK":
        val_dataset = VCTKVitsLatents(root=opt.conf_file.training.data_root, mode="val", normalize=False)

    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    val_loader = DataLoader(val_dataset,batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,)

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    if opt.conf_file["training"]["dataset"] == "LJS":
        conf_path = "ljs_base.json"
        ckpt_path = "pretrained_ljs.pth"
    else:
        conf_path = "vctk_base.json"
        ckpt_path = "pretrained_vctk.pth"

    vits_model, hps = load_vits_model(
        hps_path=os.path.join(opt.conf_file["training"]["vits_root"], "configs", conf_path),
        checkpoint_path=os.path.join(opt.conf_file["training"]["vits_root"], ckpt_path)
        )
    vits_model = vits_model.cuda().eval()

    z_to_audio = get_Z_to_audio(vits_model)
    preflow_to_audio = get_Z_preflow_to_audio(vits_model)

    if opt.txt_embeds:
        text_embedder = get_text_embedder()
        texts_man = ["recording of a man speaking"] * opt.batch_size
        texts_woman = ["recording of a woman speaking"] * opt.batch_size
        text_embeds_man = text_embedder(texts_man)
        text_embeds_woman = text_embedder(texts_woman)
        text_embeds_man = rearrange(text_embeds_man, 'b d -> b 1 d')
        text_embeds_woman = rearrange(text_embeds_woman, 'b d -> b 1 d')
    
    for loader_itr, data in enumerate(val_loader):

        if loader_itr > opt.sample_batches:
            break

        x0, x1, sids, embeds, x0_mask, x1_mask, data = runner.sample_batch(opt, val_loader, data=data)
        sids = sids.squeeze() if sids is not None else None
        
        y_masks_audio = data["y_mask_audio"]
        y_masks_text = data["y_mask_text"]
        preflow_mask = data["preflow_mask"]
        offset = data["offset"]
        audio_length = data["audio_length"]

        # generate vector for switching the sids
        sids_shuffle = torch.randperm(x0.shape[0])
        embeds_shuffled = torch.tensor([]).to(x0.device)
        for idx in sids_shuffle:
            embeds_shuffled = torch.cat((embeds_shuffled, embeds[idx].unsqueeze(0)))
        
        if opt.shuffle:
            shuffle_methods = [True, False]
        else:
            shuffle_methods = [False]
        
        if opt.txt_embeds:
            opt.shuffle = False
            shuffle_methods = [False, 'man', 'woman']

        
        for shuffling in shuffle_methods:
            # case of shuffling
            if shuffling == False:
                embs = embeds
            elif shuffling == True:
                embs = embeds_shuffled
            # case of text embeddings
            if shuffling == 'man':
                embs = text_embeds_man
            elif shuffling == 'woman':
                embs = text_embeds_woman

            with torch.no_grad():
                xs, _ = runner.ddpm_sampling(
                    opt, x1,
                    target_mask=x0_mask,
                    embeds=embs,
                    nfe=opt.nfe,
                    cfg=opt.cfg,
                    verbose=opt.global_rank==0
                )

            img_target_pred = xs[:, 0, ...]
            batch = xs.shape[0]

            # logging audio
            mse_start_pred = torch.tensor([])
            mse_pred_gt = torch.tensor([])

            for i in range(batch):
                sample = img_target_pred[i]
                gt = x0[i]
                start = x1[i]
                y_mask_text = y_masks_text[i]
                y_mask_audio = y_masks_audio[i]

                sid = torch.LongTensor([int(sids[i])]).cuda() if sids is not None else None
                audio = data["audio"][i]

                msesp = torch.nn.functional.mse_loss(start.cpu(), sample.cpu()).unsqueeze(0)
                msepgt = torch.nn.functional.mse_loss(sample.cpu(), gt.cpu()).unsqueeze(0)

                mse_start_pred = torch.cat((mse_start_pred, msesp), dim=0) if mse_start_pred.numel() > 0 else msesp
                mse_pred_gt = torch.cat((mse_pred_gt, msepgt), dim=0) if mse_pred_gt.numel() > 0 else msepgt

                # pass through vocoder
                model_audio =  z_to_audio(z=sample.cuda(), y_mask=y_mask_audio.cuda(), sid=sid).cpu().squeeze(0)

                # calculate the audio of model prediction if passed the shuffled sid too
                if shuffling == True:
                    shuffled_sid = torch.LongTensor([int(sids[sids_shuffle[i]])]).cuda()
                    model_audio_sid_shufle = z_to_audio(z=sample.cuda(), y_mask=y_mask_audio.cuda(), sid=shuffled_sid).cpu().squeeze(0)
                
                gt_audio = z_to_audio(z=gt.cuda(), y_mask=y_mask_audio.cuda(), sid=sid).cpu().squeeze(0)
                
                if opt.conf_file["training"]["z_start"] == "pre_flow":
                    start_audio = preflow_to_audio(z_p=start.cuda(), y_mask=preflow_mask[i].cuda(), sid=sid).cpu().squeeze(0)
                elif opt.conf_file["training"]["z_start"] == "post_flow":
                    start_audio = z_to_audio(z=start.cuda(), y_mask=y_masks_text[i].cuda(), sid=sid).cpu().squeeze(0)

                # cut lengths
                model_audio = model_audio[..., :audio_length[i]]
                gt_audio = gt_audio[..., :audio_length[i]]
                start_audio = start_audio[..., :audio_length[i]]
                audio = audio[..., :audio_length[i]]

                # save
                if shuffling is True:
                    permutation = f"_{sids_shuffle[i].item()}"
                elif shuffling == 'man':
                    permutation = '_man'
                elif shuffling == 'woman':
                    permutation = '_woman'
                else:
                    permutation = ""
                
                sample_path = os.path.join(RESULT_DIR, opt.conf_file.training.output_dir, f"sampling_{opt.nfe}_{opt.cfg}", f"model_audio_{i}{permutation}.wav")
                gt_path = os.path.join(RESULT_DIR, opt.conf_file.training.output_dir, f"sampling_{opt.nfe}_{opt.cfg}", f"vits_audio_{i}.wav")
                audio_path = os.path.join(RESULT_DIR, opt.conf_file.training.output_dir, f"sampling_{opt.nfe}_{opt.cfg}", f"gt_audio_{i}.wav")
                start_audio_path = os.path.join(RESULT_DIR, opt.conf_file.training.output_dir, f"sampling_{opt.nfe}_{opt.cfg}", f"start_audio_{i}.wav")

                os.makedirs(os.path.dirname(sample_path), exist_ok=True)

                save_audio(sample_path, model_audio, 22050)
                save_audio(gt_path, gt_audio, 22050)
                save_audio(audio_path, audio, 22050)
                save_audio(start_audio_path, start_audio, 22050)

                if shuffling is True:
                    sample_shuffled_path = os.path.join(RESULT_DIR, opt.conf_file.training.output_dir, f"sampling_{opt.nfe}_{opt.cfg}", f"model_audio_{i}{permutation}_vocoder_changed.wav")
                    save_audio(sample_shuffled_path, model_audio_sid_shufle, 22050)

            stats_path = '/'.join(sample_path.split('/')[:-1]) + "/stats.txt"
            log.info(f"MSE_START_PRED={mse_start_pred.mean()}\tMSE_PRED_GT={mse_pred_gt.mean()}")
            with open (stats_path, 'w', encoding='utf-8') as f:
                f.write(f"MSE_START_PRED={mse_start_pred.mean()}\tMSE_PRED_GT={mse_pred_gt.mean()}")

            dist.barrier()
    del runner
    dist.barrier()
    # sleep to avoid cuda not found errors when sampling many times
    time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")
    parser.add_argument("--model_conf_path",type=str,  default=None,        help="path to model conf file")
    parser.add_argument("--sample_batches", type=int,  default=1,           help="number of batches to sample")
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--shuffle",        action="store_true",             help="shuffle the embeddings")
    parser.add_argument("--txt_embeds",     action="store_true",             help="use text embeddings")
    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=8)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--cfg",            type=float,default=1.0,         help="CFG scale")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()
    
    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    config = yaml.load(open(opt.model_conf_path, "r"), Loader=yaml.FullLoader)
    opt.conf_file = config
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # one-time download: ADM checkpoint
    #download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
