import librosa
import torch
import os
import numpy as np
from vits import utils_diffusion
from tqdm import tqdm
from math import floor

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


    os.makedirs(os.path.join("data", f"processed_flickr/{split}"), exist_ok=True)

    def file_to_z_data(filename, root="data", data_root="processed_flickr", data_split=split):
    
        audio, sr = librosa.load(filename, sr=16000)
        duration = librosa.get_duration(audio, sr=sr)

        text_file = filename.split('/')[-1].split('.')[0] + '.txt'

        with open(os.path.join("data/flickr_audio/texts", text_file), 'r') as file:
            text = file.read()
        
        # skip long files
        if duration > 4 and skip_long:
            return
        
        # extract the embeddings
        with torch.no_grad():
            z_audio = audio_to_z(torch.tensor(audio[None, :]).cuda())["z"]
            z_text, y_mask = text_to_z(text, hps=hps)
            audio_embed = audio_embedder(audio)
            sentence_embed = sentence_embedder(text)
            phoneme_embed = phoneme_embedder(text, hps)[0]

        # save the embeddings
        file_path = os.path.join(root, data_root, data_split, filename.split('/')[-1].split('.')[0])
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

def transcribe_filelist(filelist):
    text_root = "data/flickr_audio/texts"
    os.makedirs(text_root, exist_ok=True)

    from faster_whisper import WhisperModel
    model = WhisperModel("large-v2")

    for audio_file in tqdm(filelist):
        segments, info = model.transcribe(audio_file)
        text = "".join(s.text for s in segments)

        with open(os.path.join(text_root, audio_file.split('/')[-1].split('.')[0] + '.txt'), 'w') as f:
            f.write(text)

def split_filelist(file_list, split=(0.7, 0.1, 0.2)):
    train_split, val_split, test_split = split

    train_split_index = floor(len(file_list) * train_split)
    train_split = file_list[:train_split_index]

    val_split_index = train_split_index + floor(len(file_list) * val_split)
    val_split = file_list[train_split_index:val_split_index]


    test_split = file_list[val_split_index:]

    return train_split, val_split, test_split

if __name__ == "__main__":

    splits = ['train', 'val', 'test']

    # read chunks
    root = f"data/flickr_audio/wavs"
    filelists = os.listdir(root)
    filelists = [os.path.join(root, filelist) for filelist in filelists]
    files = split_filelist(filelists, split=(0.7, 0.1, 0.2))

    for split, filelist in zip(splits, files):
        # if split == 'test':
            # transcribe_filelist(filelist)
            # print(filelist)
        process_filelist(filelist)



