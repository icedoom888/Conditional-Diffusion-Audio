from typing import Optional, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as audio_F
import torch
from torch.autograd import Variable
import auraloss
from torch import Tensor
from audio_diffusion_pytorch import UNetV0, VDiffusion, VSampler, ConditionalDiffusionVocoder, ConditionalDiffusionLLM, ConditionalDiffusionPhonemeToWav
from custom_dataset import SlidingWindow, Latent_Audio

class SpeakerEmbedder(nn.Module):
    # https://github.com/RF5/simple-speaker-embedding

    def __init__(self, sr: int):
        super().__init__()
        self.sr = sr
        self.target_sr = 16000 # needed by encoder

        model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder').to('cuda')
        model.eval()

        self.audio_embedder = model

    def forward(self, input_wav: Tensor) -> Tensor:

        input_wav = torch.squeeze(input_wav, dim = [1])
            
        # Resample input
        resampled_in = audio_F.resample(input_wav, orig_freq=self.sr, new_freq=self.target_sr)

        with torch.no_grad():
            in_embed = self.audio_embedder(resampled_in)

        return in_embed.unsqueeze(1)

class SpeakerLoss(nn.Module):
    # https://github.com/RF5/simple-speaker-embedding

    def __init__(self, sr: int):
        super().__init__()

        self.audio_embedder = SpeakerEmbedder(sr)
        self.loss = F.mse_loss

    def forward(self, input_wav: Tensor, target_wav: Tensor) -> Tensor:

        # Compute Embeddings
        in_embed = self.audio_embedder(input_wav)
        tar_embed = self.audio_embedder(target_wav)

        # Compute clap loss
        speaker_loss = self.loss(in_embed, tar_embed)

        return speaker_loss.cpu()

class CLAPLoss(nn.Module):
    # TODO: ISSUE NOT DIFFERENTIABLE. UNUSABLE
    def __init__(self, sr: int):
        import laion_clap
        super().__init__()
        self.sr = sr
        self.target_sr = 48000 # needed by CLAP encoder
        self.loss = F.mse_loss
        model = laion_clap.CLAP_Module(enable_fusion=False).cuda()
        # model.load_ckpt() # download the default pretrained checkpoint.
        self.audio_embedder = model
    
    def forward(self, input_wav: Tensor, target_wav: Tensor) -> Tensor:
        
        input_wav = torch.squeeze(input_wav, dim = [1])
        target_wav = torch.squeeze(target_wav, dim = [1])
            
        # Resample input
        resampled_in = audio_F.resample(input_wav, orig_freq=self.sr, new_freq=self.target_sr)
        # Resample output
        resampled_tar = audio_F.resample(target_wav, orig_freq=self.sr, new_freq=self.target_sr)

        with torch.no_grad():
            in_embed = self.audio_embedder.get_audio_embedding_from_data(x=resampled_in, use_tensor=True)
            tar_embed = self.audio_embedder.get_audio_embedding_from_data(x=resampled_tar, use_tensor=True)

        # Compute clap loss
        clap_loss = self.loss(in_embed, tar_embed)

        return clap_loss.cpu()

class CompositeLoss(nn.Module):
    def __init__(self, loss_args: dict, train_args: dict = None):
        self.losses = []
        self.weights = []
        self.loss_names = []

        # add all losses
        if loss_args.use_l1 == True:
            self.losses.append(F.l1_loss)
            self.weights.append(loss_args.l1_weight)
            self.loss_names.append('l1_loss')
        
        if loss_args.use_l2 == True:
            self.losses.append(F.mse_loss)
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
        
        if loss_args.use_claploss == True:
            loss_fn = CLAPLoss(sr=train_args.sr)
            self.losses.append(loss_fn)
            self.weights.append(loss_args.clap_weight)
            self.loss_names.append('clap_loss')

        if loss_args.use_speakerloss == True:
            loss_fn = SpeakerLoss(sr=train_args.sr)
            self.losses.append(loss_fn)
            self.weights.append(loss_args.speaker_weight)
            self.loss_names.append('speaker_loss')

        super(CompositeLoss, self).__init__()

    def forward(self, inputs: Tensor, targets) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss = 0
        loss_dict = {}

        for loss_fn, weight, loss_name in zip(self.losses, self.weights, self.loss_names):
            loss_item = loss_fn(inputs, targets)
            loss += weight * loss_item
            loss_dict[loss_name] = weight * loss_item

        return loss, loss_dict

def get_model(model_args, train_args, loss_fn):
    if model_args.model_type == 'ConditionalDiffusionVocoder':
        # Set up model
        model = ConditionalDiffusionVocoder(
            mel_n_fft=1024, # Mel-spectrogram n_fft
            mel_channels=192, # Mel-spectrogram channels
            mel_sample_rate=train_args.sr, # sample rate
            net_t=UNetV0,
            dim=1, # 2D U-Net working on images
            in_channels=model_args.in_channels, #IMAGE | MASK | OPTIONAL(INIT IMAGE)
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
    
    elif model_args.model_type == 'ConditionalDiffusionLLM':
        model = ConditionalDiffusionLLM(
            text_emb_channels=model_args.text_emb_channels,
            audio_emb_channels=model_args.embedding_features,
            max_len=model_args.max_wav_len,
            net_t=UNetV0,
            dim=1, # 2D U-Net working on images
            in_channels=model_args.in_channels, #IMAGE | MASK | OPTIONAL(INIT IMAGE)
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
            in_channels=model_args.in_channels, #IMAGE | MASK | OPTIONAL(INIT IMAGE)
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