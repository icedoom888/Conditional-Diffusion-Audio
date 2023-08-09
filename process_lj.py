import librosa
import torch
import os
import numpy as np
from multiprocessing import Pool
from vits import utils_diffusion

MODE = "val"

def process_filelist(filelist, skip_long=True):
    # initiate models
    text_embedder = utils_diffusion.get_text_embedder(model="CLAP")
    model, hps = utils_diffusion.load_vits_model(hps_path="vits/configs/ljs_base.json", checkpoint_path="vits/pretrained_ljs.pth")
    audio_to_z = utils_diffusion.get_audio_to_Z(model, hps)
    text_to_z = utils_diffusion.get_text_to_Z(model)

    os.makedirs(os.path.join("vits", f"processed_{MODE}"), exist_ok=True)

    def file_to_z_data(filename, root="vits", data_root=f"processed_{MODE}"):
        split = filename.split("|")
        file = "vits/" + split[0]
        filename = split[0].split('/')[-1].split('.')[0]
        text = split[1]

        audio, sr = librosa.load(file, sr=hps.data.sampling_rate)
        duration = librosa.get_duration(audio, sr=sr)
        
        # skip long files
        # if duration > 8 and skip_long:
        #  return

        # extract the embeddings
        with torch.no_grad():
            z_audio = audio_to_z(torch.tensor(audio[None, :]).cuda())["z"]
            z_text, y_mask = text_to_z(text, hps=hps, max_len=z_audio.shape[-1], y_lengths=torch.tensor([z_audio.shape[-1]]).cuda())
            text_embed = text_embedder(text)

        # save the embeddings
        file_path = os.path.join(root, data_root, filename)
        if os.path.exists(file_path):
            return
        
        np.savez_compressed(
            file_path,
            audio=audio,
            z_audio=z_audio.cpu().numpy(),
            y_mask=y_mask.cpu().numpy(),
            z_text=z_text.cpu().numpy(),
            clap_embed=text_embed.cpu().numpy()
            )
    
    for filename in filelist:
        file_to_z_data(filename)


if __name__ == "__main__":
    # read chunks
    root = f"vits/filelists/chunks_{MODE}"
    filelists = os.listdir(root)
    filelists = [os.path.join(root, filelist) for filelist in filelists]
    chunks = []
    for filelist in filelists:
        with open(filelist, "r") as f:
            chunks += [[line.strip() for line in f.readlines()]]
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with Pool(4) as p:
        p.map(process_filelist, chunks)

