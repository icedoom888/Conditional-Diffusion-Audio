import librosa
import torch
import os
import numpy as np
from multiprocessing import Pool
from vits import utils_diffusion

MODE = "val"

def process_filelist(filelist):
    # initiate models
    audio_embedder = utils_diffusion.get_audio_embedder(model="CLAP")
    model, hps = utils_diffusion.load_vits_model(hps_path="vits/configs/ljs_base.json", checkpoint_path="vits/pretrained_ljs.pth")
    audio_to_z = utils_diffusion.get_audio_to_Z(model, hps)
    text_to_z = utils_diffusion.get_text_to_Z(model)

    os.makedirs(os.path.join("vits", f"processed_ljs_{MODE}"), exist_ok=True)

    def file_to_z_data(filename, root="vits", data_root=f"processed_ljs_{MODE}"):
        split = filename.split("|")
        file = "vits/" + split[0]
        filename = split[0].split('/')[-1].split('.')[0]
        text = split[1]

        audio, sr = librosa.load(file, sr=hps.data.sampling_rate)

        # extract the embeddings
        with torch.no_grad():
            z_audio_data = audio_to_z(torch.tensor(audio[None, :]).cuda())
            z_audio, z_audio_mask = z_audio_data["z"], z_audio_data["y_mask"]
            z_text, y_mask = text_to_z(text, hps=hps, y_lengths=torch.tensor([z_audio.shape[-1]]).cuda())
            audio_48k = librosa.resample(audio, orig_sr=sr, target_sr=48000, res_type="kaiser_fast")
            audio_embed = audio_embedder(torch.tensor(audio_48k))

        # save the embeddings
        file_path = os.path.join(root, data_root, filename)
        if os.path.exists(file_path+".npz"):
            return
        
        np.savez_compressed(
            file_path,
            audio=audio,
            z_audio=z_audio.cpu().numpy(),
            z_audio_mask=z_audio_mask.cpu().numpy(),
            y_mask=y_mask.cpu().numpy(),
            z_text=z_text.cpu().numpy(),
            clap_embed=audio_embed.cpu().numpy()
            )
    
    for filename in filelist:
        file_to_z_data(filename)


if __name__ == "__main__":
    # read chunks
    root = f"vits/filelists/chunks_ljs_{MODE}"
    filelists = os.listdir(root)
    filelists = [os.path.join(root, filelist) for filelist in filelists]
    chunks = []
    for filelist in filelists:
        with open(filelist, "r") as f:
            chunks += [[line.strip() for line in f.readlines()]]
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with Pool(4) as p:
        p.map(process_filelist, chunks)

