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
import custom_dataset
from torchvision.utils import save_image
from omegaconf import OmegaConf
from audio_diffusion_pytorch import UNetV0, DiffusionVocoder, ConditionalDiffusionVocoder, ConditionalDiffusionLLM, VDiffusion, VSampler
from vits.utils_diffusion import get_audio_to_Z, get_text_to_Z, load_vits_model, get_Z_to_audio
from einops import rearrange
import wandb
from torchaudio import save as save_audio
from utils import print_sizes, CompositeLoss
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def main(conf):
    train_args = conf.training
    model_args = conf.model
    loss_args = conf.loss

    # seed alls
    set_seed(42)

    # Datasets
    train_dataset = custom_dataset.SlidingWindow(root=train_args.data_root, mode="train", sr=train_args.sr)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.train_batch_size, shuffle=False)

    # setup diffusion loss
    loss_fn = CompositeLoss(loss_args)

    # Set up model
    model = ConditionalDiffusionLLM(
        text_emb_channels=384,
        audio_emb_channels=512,
        max_len=98304,
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

    # extract the batch
    batch = next(iter(train_dataloader))

    audio = batch["audio"]
    audio_lenght = batch["audio_lenght"]
    clap_embed = batch["clap_embed"]
    sentence_embed = batch["sentence_embed"] 
    z_audio = torch.squeeze(batch["z_audio"], 1)
    z_text = torch.squeeze(batch["z_text"], 1)
    z_audio_mask = torch.squeeze(batch["z_audio_mask"], 1)
    z_text_mask = torch.squeeze(batch["z_text_mask"], 1)

    print_sizes(batch)

    # run forward 
    loss, loss_dict = model(
        audio,
        input_text_embedding=sentence_embed,
        audio_text_embedding=clap_embed,
        audio_lenght=audio_lenght,
        embedding=clap_embed,
        embedding_mask_proba=train_args.CFG_mask_proba
    )    

    print(loss, loss_dict)

    # Turn noise into new audio sample with diffusion
    model_samples = model.sample(
        input_text_embedding=sentence_embed,
        audio_text_embedding=clap_embed,
        embedding=clap_embed, # ImageBind / CLAP
        embedding_scale=1.0, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
        num_steps=50 # Higher for better quality, suggested num_steps: 10-100
    )

    # # calculate loss between samples and original
    eval_loss, eval_loss_dict = loss_fn(model_samples, audio)

    print(eval_loss, eval_loss_dict)




if __name__ == "__main__":
    # parse arguments for rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_conf.yaml")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    main(conf)
