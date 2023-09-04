from typing import Optional, Tuple, Dict
import torch.nn as nn
import torch
import auraloss
from torch import Tensor
from audio_diffusion_pytorch import UNetV0, VDiffusion, VSampler, ConditionalDiffusionVocoder, ConditionalDiffusionLLM, ConditionalDiffusionPhonemeToWav
from custom_dataset import SlidingWindow, Latent_Audio


class CompositeLoss(nn.Module):
    def __init__(self, loss_args: dict):
        self.losses = []
        self.weights = []
        self.loss_names = []

        # add all losses
        if loss_args.use_l1 == True:
            self.losses.append(nn.functional.l1_loss)
            self.weights.append(loss_args.l1_weight)
            self.loss_names.append('l1_loss')
        
        if loss_args.use_l2 == True:
            self.losses.append(nn.functional.mse_loss)
            self.weights.append(loss_args.l2_weight)
            self.loss_names.append('l2_loss')
        
        if loss_args.use_mrstft == True:
            loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
                            fft_sizes=[1024, 2048, 8192],
                            hop_sizes=[256, 512, 2048],
                            win_lengths=[1024, 2048, 8192],
                            scale="mel",
                            n_bins=128,
                            sample_rate=22050,
                            perceptual_weighting=True,
                        )
            self.losses.append(loss_fn)
            self.weights.append(loss_args.mrstft_weight)
            self.loss_names.append('mrstft_loss')

        super(CompositeLoss, self).__init__()

    def forward(self, inputs: Tensor, targets) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss = 0
        loss_dict = {}

        for loss_fn, weight, loss_name in zip(self.losses, self.weights, self.loss_names):
            loss_item = loss_fn(inputs, targets)
            loss_dict[loss_name] = loss_item

            loss += weight * loss_item

        return loss, loss_dict

def get_model(model_args, loss_fn):
    if model_args.model_type == 'ConditionalDiffusionLLM':
        model = ConditionalDiffusionLLM(
            text_emb_channels=model_args.text_emb_channels,
            audio_emb_channels=model_args.embedding_features,
            max_len=model_args.max_wav_len,
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
    
    elif model_args.model_type == 'ConditionalDiffusionPhonemeToWav':
        model = ConditionalDiffusionPhonemeToWav(
            text_emb_channels=model_args.text_emb_channels,
            audio_emb_channels=model_args.embedding_features,
            max_len=model_args.max_wav_len,
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
    
    else:
        print(f"Undefined model with type: {model_args.model_type}")
        raise ValueError

    return model

def get_datasets(train_args):
    if train_args.dataset_name == 'SlidingWindow':
        print(f'Building dataset: {train_args.dataset_name}')
        train_dataset = SlidingWindow(root=train_args.data_root, mode="train", sr=train_args.sr)
        val_dataset = SlidingWindow(root=train_args.data_root, mode="val", sr=train_args.sr)
    
    elif train_args.dataset_name == 'Latent_Audio':
        print(f'Building dataset: {train_args.dataset_name}')
        train_dataset = Latent_Audio(root=train_args.data_root, mode="train")
        val_dataset = Latent_Audio(root=train_args.data_root, mode="val")
    
    else:
        print(f"Undefined dataset with name: {train_args.dataset_name}")
        raise ValueError
    
    return train_dataset, val_dataset