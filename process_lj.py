import librosa
import torch
import os
import numpy as np
from vits import utils_diffusion
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"]='true'

def process_filelist(filelist, skip_long=True):
    # initiate models
    audio_embedder = utils_diffusion.get_audio_embedder(model="CLAP")
    model, hps = utils_diffusion.load_vits_model(hps_path="vits/configs/ljs_base.json", checkpoint_path="vits/pretrained_ljs.pth")
    model = model.cuda()
    model = model.eval()
    audio_to_z = utils_diffusion.get_audio_to_Z(model, hps)
    text_to_z = utils_diffusion.get_text_to_Z(model)
    sentence_embedder = utils_diffusion.get_sentence_embedder(model='all-MiniLM-L6-v2')
    phoneme_embedder = utils_diffusion.get_phoneme_embedder(model)

    os.makedirs(os.path.join("data", f"processed_LJSpeech/{split}"), exist_ok=True)

    def file_to_z_data(filename, root="data", data_root="processed_LJSpeech", data_split=split):
        split = filename.split("|")
        file = "data/LJSpeech-1.1/" + split[0]
        filename = split[0].split('/')[-1].split('.')[0]
        text = split[1]

        audio, sr = librosa.load(file, sr=hps.data.sampling_rate)
        duration = librosa.get_duration(audio, sr=sr)
        
        # skip long files
        if duration > 4 and skip_long:
            return
        
        # extract the embeddings
        with torch.no_grad():
            z_audio = audio_to_z(torch.tensor(audio[None, :]).cuda())["z"]
            z_text, y_mask = text_to_z(text, hps=hps)

            audio_48k = librosa.resample(audio, orig_sr=sr, target_sr=48000, res_type="kaiser_fast")
            audio_embed = audio_embedder(torch.tensor(audio_48k))
            sentence_embed = sentence_embedder(text)
            phoneme_embed = phoneme_embedder(text, hps)[0]

        # save the embeddings
        file_path = os.path.join(root, data_root, data_split, filename)
        if os.path.exists(file_path):
            return
        
        np.savez_compressed(
            file_path,
            audio=audio,
            z_audio=z_audio.cpu().numpy(),
            y_mask=y_mask.cpu().numpy(),
            z_text=z_text.cpu().numpy(),
            clap_embed=audio_embed.cpu().numpy(),
            sentence_embed=sentence_embed.cpu().numpy(),
            phoneme_embed=phoneme_embed.cpu().numpy(),
            )
    
    for filename in tqdm(filelist):
        file_to_z_data(filename, data_split=split)


if __name__ == "__main__":
    # read chunks
    splits = ['train', 'test', 'val']
    for split in splits:

        root = f"vits/filelists/chunks_{split}"
        filelists = os.listdir(root)
        filelists = [os.path.join(root, filelist) for filelist in filelists]
        chunks = []
        for filelist in filelists:
            with open(filelist, "r") as f:
                chunks += [[line.strip() for line in f.readlines()]]

        process_filelist(chunks[0])
