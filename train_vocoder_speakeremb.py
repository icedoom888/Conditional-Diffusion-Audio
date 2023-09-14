# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py
import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, set_seed
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import wandb
from torchaudio import save as save_audio
from utils import CompositeLoss, SpeakerEmbedder, get_model, get_datasets
from funcs import print_sizes
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def main(conf):
    train_args = conf.training
    model_args = conf.model
    loss_args = conf.loss

    # create working directory
    output_dir = train_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    # seed alls
    set_seed(42)

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)],
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=train_args.mixed_precision,
        log_with="wandb"
    )

    # Get datasets and dataloaders
    train_dataset, val_dataset = get_datasets(train_args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.train_batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_args.eval_batch_size, shuffle=True)

    # setup diffusion loss
    loss_fn = CompositeLoss(loss_args, train_args)

    # Set up model
    model = get_model(model_args, train_args, loss_fn)

    # Set up speaker embedding
    speaker_embedder = SpeakerEmbedder(train_args.sr)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

    # Set up scheduler
    lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * train_args.num_epochs) //
        train_args.gradient_accumulation_steps,
    )

    # Accelerate
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    global_step = 0
    
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
        print(train_args.wandb_project)
        accelerator.init_trackers(
            project_name=train_args.wandb_project,
            config=OmegaConf.to_container(conf, resolve=True),
            init_kwargs={"wandb": {"entity": "ethz-mtc", "group": "conditional-vocoder"}}
            )
    
    for epoch in range(train_args.start_epoch, train_args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), initial=epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        
        for step, batch in enumerate(train_dataloader):

            # extract the batch
            audio = batch["audio"]
            z_audio = batch["z_audio"]
            z_audio_mask = batch["z_audio_mask"]
            z_text = batch["z_text"]
            z_text_mask = batch["z_text_mask"]

            # Compute speaker embedding 
            embeds = speaker_embedder(audio)

            # get the latents Z 
            input_spec = z_audio
            
            with accelerator.accumulate(model):
                loss, loss_dict = model(
                    audio,
                    input_spec=input_spec,
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

                # extract the batch
                audio = eval_batch["audio"]
                z_audio = eval_batch["z_audio"]
                z_audio_mask = eval_batch["z_audio_mask"]
                z_text = eval_batch["z_text"]
                z_text_mask = eval_batch["z_text_mask"]

                # Compute speaker embedding 
                embeds = speaker_embedder(audio)
                
                # Turn noise into new audio sample with diffusion
                model_samples = model.sample(
                    spectrogram=z_text.unsqueeze(1),
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
                    accelerator.log({"audio_examples":
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
                accelerator.log(eval_logs, step=global_step)


        accelerator.wait_for_everyone()

    accelerator.end_training()



if __name__ == "__main__":
    # parse arguments for rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_conf.yaml")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    main(conf)
