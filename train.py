# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py
import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, set_seed
from tqdm.auto import tqdm
import custom_dataset
from torchvision.utils import save_image
from omegaconf import OmegaConf
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from vits.utils_diffusion import get_audio_to_Z, get_text_to_Z, load_vits_model, get_Z_to_audio
from einops import rearrange
import wandb
from torchaudio import save as save_audio

logger = get_logger(__name__)

def main(conf):
    train_args = conf.training
    model_args = conf.model

    # create working directory
    output_dir = train_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    # seed alls
    set_seed(42)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=train_args.mixed_precision,
        log_with="wandb"
    )

    train_dataset = custom_dataset.LJS_Latent(root=train_args.data_root, mode="train")
    val_dataset = custom_dataset.LJS_Latent(root=train_args.data_root, mode="val")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.train_batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_args.eval_batch_size, shuffle=True)

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

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    global_step = 0
    
    # set seeds


    # load weights
    if os.path.exists(os.path.join(output_dir, f"model_latest.pt")):
        map_location = {"cuda:%d" % 0: "cuda:%d" % accelerator.process_index}
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

    if accelerator.is_main_process:
        # initialize wandb
        accelerator.init_trackers(
            project_name=train_args.wandb_project, 
            config=OmegaConf.to_container(conf, resolve=True),
            init_kwargs={"wandb": {"entity": "matvogel"}}
            )
        # initialize vits functions
        vits_model, hps = load_vits_model(hps_path="vits/configs/ljs_base.json", checkpoint_path="vits/pretrained_ljs.pth")
        z_to_audio = get_Z_to_audio(vits_model)
    
    for epoch in range(train_args.start_epoch, train_args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), initial=epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        
        for step, batch in enumerate(train_dataloader):
            # extract the batch
            z_audio = batch["z_audio"]
            z_audio_mask = batch["z_audio_mask"]
            z_text = batch["z_text"]
            z_text_mask = batch["z_text_mask"]
            embeds = batch["clap_embed"]

            # process the pair to get the latents Z and the embeddings
            init_image = z_audio_mask
            if model_args.use_initial_image:
                init_image = torch.cat([init_image, z_text], dim=1)
            
            with accelerator.accumulate(model):
                loss = model(
                    z_audio,
                    features = z_text.mean(-3) if model_args.use_additional_time_conditioning else None,
                    init_image=init_image,
                    embedding=embeds,
                    embedding_mask_proba=train_args.CFG_mask_proba
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if train_args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)

            if (global_step) % train_args.wandb_log_every == 0 or global_step == 1:
                wandb_log = logs.copy()
                wandb_log.pop("step")
                accelerator.log(wandb_log, step=global_step)

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if ((epoch + 1) % train_args.save_model_epochs == 0
                    or (epoch + 1) % train_args.save_images_epochs == 0
                    or epoch == train_args.num_epochs - 1):
                unet = accelerator.unwrap_model(model)
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

                z_audio = eval_batch["z_audio"]
                z_audio_mask = eval_batch["z_audio_mask"]
                y_mask = eval_batch["y_mask"]
                z_text = eval_batch["z_text"]
                z_text_mask = eval_batch["z_text_mask"]
                embeds = eval_batch["clap_embed"]
                # masking information
                offset = eval_batch["offset"]
                z_length = eval_batch["z_audio_length"]

                # Turn noise into new audio sample with diffusion
                initial_noise = torch.randn_like(z_audio)
                # append mask
                init_image = z_audio_mask
                # append initial image
                if model_args.use_initial_image:
                    init_image = torch.cat([init_image, z_text], dim=1)
                
                model_samples = model.sample(
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
                save_image(batch_images, os.path.join(output_dir, "samples", f"model_samples_{epoch}.png"))
                save_image(batch_gt, os.path.join(output_dir, "samples", f"gt_samples_{epoch}.png"))
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

                for i in range(model_samples.shape[0]):
                    sample = model_samples[i]
                    gt = z_audio[i]
                    mask = y_mask[i]

                    # cut padding
                    sample = sample[:, :, offset[i]:offset[i]+z_length[i]]
                    gt = gt[:, :, offset[i]:offset[i]+z_length[i]]
                    mask = mask[:, offset[i]:offset[i]+z_length[i]]

                    # pass through vocoder
                    model_audio =  z_to_audio(z=sample, y_mask=mask).cpu().squeeze(0)
                    gt_audio = z_to_audio(z=gt, y_mask=mask).cpu().squeeze(0)

                    # save
                    sample_path = os.path.join(output_dir, "samples", f"model_audio_{epoch}_{i}.wav")
                    gt_path = os.path.join(output_dir, "samples", f"gt_audio_{epoch}_{i}.wav")
                    save_audio(sample_path, model_audio, 22050)
                    save_audio(gt_path, gt_audio, 22050)

                    # log to wandb
                    accelerator.log({"audio_examples":
                                     [
                                        wandb.Audio(sample_path, caption=f"Sample {i}", sample_rate=22050),
                                        wandb.Audio(gt_path, caption=f"Ground Truth {i}", sample_rate=22050)
                                     ]}, step=global_step)

                # log to wandb
                accelerator.log({"eval_loss": eval_loss.detach().item()}, step=global_step)
                accelerator.log({"eval_images": [images_model, images_gt]}, step=global_step)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    # parse arguments for rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--config", type=str, default="configs/train_conf.yaml")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    # setup rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    conf.local_rank = args.local_rank

    main(conf)
