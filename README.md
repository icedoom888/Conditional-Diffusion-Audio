# Conditional-Diffusion-Audio

## Prerequisites
- Python >= 3.7
- Pytorch with cuda (tested on 1.13)
- VITS installation including monotonic_align

## Usage VITS functions
There are additional functions in `vits/utils_diffusion.py` to extract the latents Z from VITS and to generate audio from the latents Z.

```python	
from vits.utils_diffusion import get_audio_to_Z, get_text_to_Z, load_vits_model, get_Z_to_audio

# build the vits model, returns the model and the hparams dict
vits_model, vits_hp = load_vits_model(
    hps_path="vits/configs/ljs_base.json", # path to the hparams file
    device=accelerator.device, # torch device
    checkpoint_path="vits/pretrained_ljs.pth") # path to the pretrained model

# VITS functions preparation
audio_to_Z = get_audio_to_Z(vits_model, vits_hp)
text_to_Z = get_text_to_Z(vits_model)
Z_to_audio = get_Z_to_audio(vits_model)

# usage of functions
data = audio_to_z(audio_tensor) # audio tensor has shape C X T
z = data["z"] # data contains z and y_mask, used for z to audio (and more)

audio = z_to_audio(z, data["y_mask"])

z, y_mask = text_to_z("Hello world.", hps=vits_hp) # returns z and y_mask directly

```

## Oldscool Img2Img
The code is in the root directory in `train.py`. It generates a UNET and Diffusion pipeline with audio-diffusion-pytorch and uses their loss calculation function. The training script is from Huggingface Mel Diffusion.

### Installation
```bash
pip install -e ./audio-diffusion-pytorch
pip install -e ./a-unet
pip install -r requirements.txt
```

To run it generate a accelerate configuration and use `sh train_accelerate.sh`. For wandb run `wandb login` in the console.

## I2SB
Code is in I2SB.

## TODO
- add sample script (with CFG)
- add evaluation loss logging to wandb
- generate bigger dataset
- content based losses for traditional approaches
- https://github.com/csteinmetz1/auraloss

