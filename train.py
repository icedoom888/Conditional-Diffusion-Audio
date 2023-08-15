# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py
import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.training_utils import set_seed
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm
import custom_dataset
from torchvision.utils import save_image
from omegaconf import OmegaConf
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from vits.utils_diffusion import get_audio_to_Z, get_text_to_Z, load_vits_model, get_Z_to_audio
from einops import rearrange
import wandb
from torchaudio import save as save_audio
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from I2SB.logger import Logger

logger = get_logger(__name__)


def train(gpu, conf):
    train_args = conf.training
    model_args = conf.model

    log = Logger(rank=gpu, log_dir="logging")

    # create working directory
    output_dir = train_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    # seed alls
    set_seed(42)

    if conf.DDP:
        conf.rank = gpu
        dist.init_process_group(
            backend='nccl',
            init_method=conf.dist_url,
            world_size=conf.world_size,
            rank=conf.rank
        )
        torch.cuda.set_device(gpu)         

    # setup dataset specific parameters
    if train_args.dataset == "LJS":
        train_dataset = custom_dataset.LJSSlidingWindow(root=train_args.data_root, mode="train", normalize=False)
        val_dataset = custom_dataset.LJSSlidingWindow(root=train_args.data_root, mode="val", normalize=False)
        data_mean = custom_dataset.LJS_MEAN_TEXT
        data_std = custom_dataset.LJS_STD_TEXT
        conf_path = "vits/configs/ljs_base.json"
        ckpt_path = "vits/pretrained_ljs.pth"
    elif train_args.dataset == "VCTK":
        train_dataset = custom_dataset.VCTKSlidingWindow(root=train_args.data_root, mode="train", normalize=False)
        val_dataset = custom_dataset.VCTKSlidingWindow(root=train_args.data_root, mode="val", normalize=False)
        data_mean = custom_dataset.VCTK_MEAN_TEXT
        data_std = custom_dataset.VCTK_STD_TEXT
        conf_path = "vits/configs/vctk_base.json"
        ckpt_path = "vits/pretrained_vctk.pth"
    else:
        raise NotImplementedError("Dataset not implemented!")

    if conf.DDP:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=conf.world_size, rank=conf.rank)
    else:
        sampler = None
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.micro_batch_size, shuffle=False, num_workers=4, sampler=sampler)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_args.eval_batch_size, shuffle=False, num_workers=4, sampler=sampler)

    # setup diffusion loss
    if train_args.loss_fn_diffusion == "l1":
        loss_fn = torch.nn.functional.l1_loss
    elif train_args.loss_fn_diffusion == "l2":
        loss_fn = torch.nn.functional.mse_loss
    else:
        raise NotImplementedError

    model = DiffusionModel(
        net_t=UNetV0,
        dim=2, # 2D U-Net working on images
        in_channels=2 if not model_args.use_initial_image else 3, #IMAGE | MASK | OPTIONAL(INIT IMAGE)
        out_channels = 1, # 1 for the output image
        channels=list(model_args.channels), # U-Net: number of channels per layer
        factors=list(model_args.factors), # U-Net: image size reduction per layer
        items=list(model_args.layers), # U-Net: number of layers
        attentions=list(model_args.attentions), # U-Net: number of attention layers
        cross_attentions=list(model_args.cross_attentions), # U-Net: number of cross attention layers
        attention_heads=model_args.attention_heads, # U-Net: number of attention heads per attention item
        attention_features=model_args.attention_features , # U-Net: number of attention features per attention item
        diffusion_t=VDiffusion, # The diffusion method used
        sampler_t=VSampler, # The diffusion sampler used
        loss_fn=loss_fn, # The loss function used
        use_text_conditioning=False, # U-Net: enables text conditioning (default T5-base)
        use_additional_time_conditioning=model_args.use_additional_time_conditioning, # U-Net: enables additional time conditionings
        use_embedding_cfg=model_args.use_embedding_cfg, # U-Net: enables classifier free guidance
        embedding_max_length=model_args.embedding_max_length, # U-Net: text embedding maximum length (default for T5-base)
        embedding_features=model_args.embedding_features, # text embedding dimensions, is used for CFG
    )

    # setup diffusion parameters
    model.diffusion.randn_mean = data_mean
    model.diffusion.randn_std = data_std

    # move to right device
    model = model.to(gpu)

    ema = ExponentialMovingAverage(model.net.parameters(), decay=train_args.ema_max_decay)
    ema.to(gpu)

    if conf.DDP:
        model = DDP(model, device_ids=[gpu])

    if train_args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_args.lr_warmup_steps,
        num_training_steps=train_args.num_train_steps // (train_args.train_batch_size // train_args.micro_batch_size),
    )

    start_step = 0

    # load weights
    if os.path.exists(os.path.join(output_dir, f"model_latest.pt")):
        log.info("Loading weights...")
        map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
        state_dicts = torch.load(os.path.join(output_dir, f"model_latest.pt"), map_location=map_location)
        model.load_state_dict(state_dicts["model"])
        if "ema" in state_dicts.keys():
            ema.load_state_dict(state_dicts["ema"])
        optimizer.load_state_dict(state_dicts["optimizer"])
        lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
        start_step = state_dicts["global_step"]
        log.info("Weights successfully loaded...")

    if gpu == 0:
        # initialize wandb
        wandb.init(project=train_args.wandb_project, entity="ethz-mtc", config=OmegaConf.to_container(conf, resolve=True))
        # initialize vits functions
        vits_model, hps = load_vits_model(hps_path=conf_path, checkpoint_path=ckpt_path)
        z_to_audio = get_Z_to_audio(vits_model)
        # export master ip to MASTER_ADDR
    
    n_inner_loop = train_args.train_batch_size // (conf.world_size * train_args.micro_batch_size)
    train_iter = custom_dataset.iterate_loader(train_dataloader)
    val_iter = custom_dataset.iterate_loader(val_dataloader)

    model.train()

    for global_step in range(start_step, int(train_args.num_train_steps)):
        #progress_bar = tqdm(total=len(train_dataloader), initial=epoch, disable=not gpu == 0)
        #progress_bar.set_description(f"Epoch {epoch}")
        optimizer.zero_grad()
        
        # do gradient accumulations
        for _ in range(n_inner_loop):
            batch = next(train_iter)
    
            # extract the batch
            z_audio = batch["z_audio"].to(gpu)
            z_audio_mask = batch["z_audio_mask"].to(gpu)
            z_text = batch["z_text"].to(gpu)
            z_text_mask = batch["z_text_mask"].to(gpu)
            embeds = batch["clap_embed"].to(gpu)
            
            # process the pair to get the latents Z and the embeddings
            init_image = z_audio_mask
            if model_args.use_initial_image:
                init_image = torch.cat([init_image, z_text], dim=1)
            with torch.cuda.amp.autocast(enabled=not scaler is None):
                loss = model(
                    z_audio,
                    features = z_text.mean(-3) if model_args.use_additional_time_conditioning else None,
                    init_image=init_image,
                    embedding=embeds,
                    embedding_mask_proba=train_args.CFG_mask_proba
                )

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        lr_scheduler.step()
        if train_args.use_ema:
            ema.update()

        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }

        if (global_step + 1) % train_args.log_every == 0:
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+global_step,
                int(train_args.num_train_steps),
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))

            if gpu==0:
                wandb_log = logs.copy()
                wandb_log.pop("step")
                wandb.log(wandb_log, step=global_step+1)
        
        # Generate sample images for visual inspection
        if gpu==0:
            log_step = global_step + 1
            if log_step % train_args.save_every == 0:
                log.info("Saving model...")
                save_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "global_step": global_step,
                }
                torch.save(save_data, os.path.join(output_dir, f"model_latest.pt"))
                if log_step % 1000 == 0:
                    torch.save(save_data, os.path.join(output_dir, f"model_{log_step}.pt"))

            if log_step % train_args.eval_every == 0:
                log.info("Sampling...")
                eval_batch = next(val_iter)
                # put everything on device
                for k, v in eval_batch.items():
                    eval_batch[k] = v.to(gpu)

                z_audio = eval_batch["z_audio"]
                z_audio_mask = eval_batch["z_audio_mask"]
                y_mask = eval_batch["y_mask"]
                z_text = eval_batch["z_text"]
                z_text_mask = eval_batch["z_text_mask"]
                embeds = eval_batch["clap_embed"]
                # masking information
                offset = eval_batch["offset"]
                z_length = eval_batch["z_audio_length"]
                # sid for multi speaker ds
                if "sid" in batch.keys():
                    sids = batch["sid"]
                else:
                    sids = None

                # Turn noise into new audio sample with diffusion
                initial_noise = torch.normal(mean=data_mean, std=data_std, size=z_audio.shape).to(gpu)
                # append mask
                init_image = z_audio_mask
                # append initial image
                if model_args.use_initial_image:
                    init_image = torch.cat([init_image, z_text], dim=1)
                
                if conf.DDP:
                    sample_model = model.model
                else:
                    sample_model = model
                
                model_samples = sample_model.sample(
                    initial_noise, # NOISE | MASK | OPTIONAL(INIT IMAGE)
                    init_image=init_image,
                    features = z_text.mean(-3) if model_args.use_additional_time_conditioning else None,
                    embedding=embeds, # ImageBind / CLAP
                    embedding_scale=1.0, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
                    num_steps=50 # Higher for better quality, suggested num_steps: 10-100
                )

                # calculate loss between samples and original
                eval_loss =loss_fn(model_samples, z_audio)

                batch_images = rearrange(model_samples, "b c h w -> c h (b w)")
                batch_gt = rearrange(z_audio, "b c h w -> c h (b w)")
                # scale to 0-1
                batch_images = custom_dataset.scale_0_1(batch_images)
                batch_gt = custom_dataset.scale_0_1(batch_gt)
                # save locally
                #save_image(batch_images, os.path.join(output_dir, "samples", f"model_samples_{log_step}.png"))
                #save_image(batch_gt, os.path.join(output_dir, "samples", f"gt_samples_{log_step}.png"))
                # scale to 0-255
                batch_images = batch_images * 255
                batch_gt = batch_gt * 255
                # clamp to 0-255
                batch_images = batch_images.clamp(0, 255).long()
                batch_gt = batch_gt.clamp(0, 255).long()
                # create wandb images
                batch_images = rearrange(batch_images, "c h w -> h w c").cpu().numpy()
                batch_gt = rearrange(batch_gt, "c h w -> h w c").cpu().numpy()
                images_model = wandb.Image(batch_images, caption="Model Samples")
                images_gt = wandb.Image(batch_gt, caption="Ground Truth")

                msg = f"Saving {model_samples.shape[0]} samples..."
                log.info(msg)

                for i in range(model_samples.shape[0]):
                    sample = model_samples[i]
                    gt = z_audio[i]
                    mask = y_mask[i]
                    if sids is not None:
                        sid = torch.LongTensor([int(sids[i])]).cuda()
                    else:
                        sid = None
                    
                    # cut padding
                    sample = sample[:, :, offset[i]:offset[i]+z_length[i]]
                    gt = gt[:, :, offset[i]:offset[i]+z_length[i]]
                    mask = mask[:, offset[i]:offset[i]+z_length[i]]

                    # pass through vocoder
                    model_audio =  z_to_audio(z=sample, y_mask=mask, sid=sid).cpu().squeeze(0)
                    gt_audio = z_to_audio(z=gt, y_mask=mask, sid=sid).cpu().squeeze(0)

                    # save
                    sample_path = os.path.join(output_dir, "samples", f"model_audio_{log_step}_{i}.wav")
                    gt_path = os.path.join(output_dir, "samples", f"gt_audio_{log_step}_{i}.wav")
                    save_audio(sample_path, model_audio, 22050)
                    save_audio(gt_path, gt_audio, 22050)

                    # log to wandb
                    wandb.log({"audio_examples":
                                     [
                                        wandb.Audio(sample_path, caption=f"Sample {i}", sample_rate=22050),
                                        wandb.Audio(gt_path, caption=f"Ground Truth {i}", sample_rate=22050)
                                     ]}, step=log_step)

                # log to wandb
                wandb.log({"eval_loss": eval_loss.detach().item()}, step=log_step)
                wandb.log({"eval_images": [images_model, images_gt]}, step=log_step)
                log.info("Sampling finished...")
        
        if conf.DDP:
            dist.barrier()
        
    wandb.finish()


if __name__ == "__main__":
    # parse arguments for rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_conf.yaml")
    parser.add_argument("--DDP", action="store_true", default=False)
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    conf.DDP = args.DDP

    DIST_FILE = "ddp_sync_"

    ngpus = torch.cuda.device_count()
    print("Number of GPUs: {}".format(ngpus))

    # setup for DDP
    if conf.DDP:
        conf.gpus = ngpus
        conf.world_size = conf.gpus
        job_id = os.environ["SLURM_JOBID"]
        conf.dist_url = "file://{}.{}".format(os.path.realpath(DIST_FILE), job_id)
    else:
        conf.world_size = ngpus

    if conf.DDP and ngpus > 1:
        mp.spawn(train, nprocs=conf.gpus, args=(conf,))
    else:
        train(0, conf)
