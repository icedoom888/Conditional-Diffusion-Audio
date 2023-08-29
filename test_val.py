# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py
import argparse
import os
import torch
# torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.utils import data
from torch import distributed as dist
from torch import nn, autograd, optim

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, set_seed
from tqdm.auto import tqdm
import custom_dataset
from torchvision.utils import save_image
from omegaconf import OmegaConf
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler, DiffusionVocoder, ConditionalDiffusionVocoder
from vits.utils_diffusion import get_audio_to_Z, get_text_to_Z, load_vits_model, get_Z_to_audio
from einops import rearrange
import wandb
from torchaudio import save as save_audio
from utils import print_sizes, CompositeLoss
import warnings
warnings.filterwarnings("ignore")

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def main(conf):
    train_args = conf.training
    model_args = conf.model
    loss_args = conf.loss

    print(args)

    # create working directory
    output_dir = train_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    # seed alls
    set_seed(42)

    if train_args.dataset_name == 'SlidingWindow':
        print(f'Building dataset: {train_args.dataset_name}')
        train_dataset = custom_dataset.SlidingWindow(root=train_args.data_root, mode="train", sr=train_args.sr)
        val_dataset = custom_dataset.SlidingWindow(root=train_args.data_root, mode="val", sr=train_args.sr)
    
    elif train_args.dataset_name == 'Latent_Audio':
        print(f'Building dataset: {train_args.dataset_name}')
        train_dataset = custom_dataset.Latent_Audio(root=train_args.data_root, mode="train")
        val_dataset = custom_dataset.Latent_Audio(root=train_args.data_root, mode="val")
    
    else:
        raise ValueError

    args.n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_args.train_batch_size, 
                                                   sampler=data.distributed.DistributedSampler(train_dataset, shuffle=False))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=train_args.eval_batch_size, 
                                                 sampler=data.distributed.DistributedSampler(val_dataset, shuffle=False))

    # setup diffusion loss
    loss_fn = CompositeLoss(loss_args)

    # Set up model
    model = ConditionalDiffusionVocoder(
        mel_n_fft=1024, # Mel-spectrogram n_fft
        mel_channels=192, # Mel-spectrogram channels
        mel_sample_rate=train_args.sr, # sample rate
        net_t=UNetV0,
        dim=1, # 2D U-Net working on images
        in_channels=3, #IMAGE | MASK | OPTIONAL(INIT IMAGE)
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
    model.to('cuda')

    # Multi-GPU
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

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
        num_training_steps=(len(train_dataloader) * train_args.num_epochs) //
        train_args.gradient_accumulation_steps,
    )

    global_step = 0
    # load weights
    if os.path.exists(os.path.join(output_dir, f"model_latest.pt")):
        map_location = "cpu"
        state_dicts = torch.load(os.path.join(output_dir, f"model_latest.pt"), map_location=map_location)
        model.load_state_dict(state_dicts["model"])
        optimizer.load_state_dict(state_dicts["optimizer"])
        lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
        train_args.start_epoch = state_dicts["epoch"] + 1
        global_step = state_dicts["global_step"]
    
    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=train_args.ema_inv_gamma,
        power=train_args.ema_power,
        max_value=train_args.ema_max_decay,
    )

    if global_step > 0:
        ema_model.optimization_step = global_step

    # Pass the config dictionary when you initialize W&B
    run = wandb.init(
        entity="ethz-mtc",
        group="conditional-vocoder",
        project=train_args.wandb_project
    )

    for epoch in range(train_args.start_epoch, train_args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), initial=epoch)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):

            # extract the batch
            audio = batch["audio"]
            z_audio = batch["z_audio"]
            z_audio_mask = batch["z_audio_mask"]
            z_text = batch["z_text"]
            z_text_mask = batch["z_text_mask"]
            embeds = batch["clap_embed"]

            if train_args.dataset_name == 'SlidingWindow':
                z_audio = torch.squeeze(z_audio, 1)
                z_text = torch.squeeze(z_text, 1)
                z_audio_mask = torch.squeeze(z_audio_mask, 1)
                z_text_mask = torch.squeeze(z_text_mask, 1)


            # process the pair to get the latents Z and the embeddings
            input_spec = z_audio
            
            loss, loss_dict = model(
                audio,
                input_spec=input_spec,
                embedding=embeds,
                embedding_mask_proba=train_args.CFG_mask_proba
            )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if train_args.use_ema:
                ema_model.step(model)
            optimizer.zero_grad()

            progress_bar.update(1)

            # update the step only if gradient was updated
            if step % train_args.gradient_accumulation_steps == 0:
                global_step += 1

            # Save logs
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }

            for loss_name in loss_dict.keys():
                logs[loss_name] = loss_dict[loss_name].detach().item()

            progress_bar.set_postfix(**logs)

            if (global_step) % train_args.wandb_log_every == 0 or global_step == 1:
                wandb_log = logs.copy()
                wandb_log.pop("step")
                wandb.log(wandb_log, step=global_step)
        
        progress_bar.close()

        # Generate sample images for visual inspection
        if get_rank() == 0:
            if ((epoch + 1) % train_args.save_model_epochs == 0
                    or (epoch + 1) % train_args.save_images_epochs == 0
                    or epoch == train_args.num_epochs - 1):
                unet = model
                if train_args.use_ema:
                    ema_model.copy_to(unet.parameters())

            if (epoch + 1) % train_args.save_model_epochs == 0 or epoch == train_args.num_epochs - 1:
                save_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }
                torch.save(save_data, os.path.join(output_dir, f"model_latest.pt"))

            if (epoch + 1) % train_args.save_images_epochs == 0:
                eval_batch = next(iter(val_dataloader))

                # extract the batch
                audio = eval_batch["audio"]
                z_audio = eval_batch["z_audio"]
                z_audio_mask = eval_batch["z_audio_mask"]
                z_text = eval_batch["z_text"]
                z_text_mask = eval_batch["z_text_mask"]
                embeds = eval_batch["clap_embed"]

                if train_args.dataset_name == 'SlidingWindow':
                    z_audio = torch.squeeze(z_audio, 1)
                    z_text = torch.squeeze(z_text, 1)
                    z_audio_mask = torch.squeeze(z_audio_mask, 1)
                    z_text_mask = torch.squeeze(z_text_mask, 1)

                # print_sizes(eval_batch)
                
                # Turn noise into new audio sample with diffusion
                with torch.no_grad():
                    model_samples = model.module.sample(
                        spectrogram=z_text.unsqueeze(1).cuda(),
                        embedding=embeds, # ImageBind / CLAP
                        embedding_scale=1.0, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
                        num_steps=50 # Higher for better quality, suggested num_steps: 10-100
                    )

                    # calculate loss between samples and original
                    eval_loss, eval_loss_dict = loss_fn(model_samples, audio)

                for i in range(model_samples.shape[0]):
                    sample = model_samples[i]

                    # save
                    sample_path = os.path.join(output_dir, "samples", f"model_audio_{epoch}_{i}.wav")
                    gt_path = os.path.join(output_dir, "samples", f"gt_audio_{epoch}_{i}.wav")

                    save_audio(sample_path, sample.cpu(), train_args.sr)
                    save_audio(gt_path, audio[i].cpu(), train_args.sr)

                    # log to wandb
                    wandb.log({"audio_examples":
                                     [
                                        wandb.Audio(sample_path, caption=f"Sample {i}", sample_rate=train_args.sr),
                                        wandb.Audio(gt_path, caption=f"Ground Truth {i}", sample_rate=train_args.sr)
                                     ]}, step=global_step)
                # Save logs
                eval_logs = {
                    "eval_loss": eval_loss.detach().item(),
                }

                for loss_name in eval_loss_dict.keys():
                    eval_logs[loss_name] = eval_loss_dict[loss_name].detach().item()

                # log to wandb
                wandb.log(eval_logs, step=global_step)



if __name__ == "__main__":

    # Set available GPUS
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


    # parse arguments for rank
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument("--config", type=str, default="configs/train_conf.yaml")

    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    main(conf)
