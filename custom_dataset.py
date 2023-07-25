from torch.utils.data import Dataset
import torch
import os
import numpy as np
from torchvision.transforms import Normalize


# stats for normalization
Z_AUDIO_MEAN = -0.0270
Z_AUDIO_STD = 1.0935
Z_TEXT_MEAN = -0.0277
Z_TEXT_STD = 1.1185


def scale_0_1(x, eps=1e-7):
    return (x - x.min()) / (x.max() - x.min() + eps)


def scale_01_to_11(x):
    normalize = Normalize(mean=[0.5], std=[0.5])
    return normalize(x)


class DummyDataset(Dataset):
    def __init__(self, root=None, transform=None):
        self.transform = transform

    def __getitem__(self, index):
        data = dict(
            source=torch.randn(1, 192, 320),
            target=torch.randn(1, 192, 320),
            text_embeds=torch.randn(32, 768)
        )
        return data

    def __len__(self):
        return 100


class LJS_Latent(Dataset):
    def __init__(self, root, mode="train", max_len_seq=384):
        super().__init__()
        self.root = root
        self.mode = mode
        self.max_len_seq = max_len_seq
        self.load_data()
        self.normalize_z_audio = Normalize(mean=[Z_AUDIO_MEAN], std=[Z_AUDIO_STD])
        self.normalize_z_text = Normalize(mean=[Z_TEXT_MEAN], std=[Z_TEXT_STD])

    def load_data(self):
        self.data = os.listdir(os.path.join(self.root, self.mode))
        assert len(self.data) > 0, "No data found"
    
    def pad_and_shift(self, data, random_offset=None):
        canvas = torch.zeros((data.shape[-2], self.max_len_seq))
        mask = canvas.clone()
        canvas[:, random_offset:random_offset+data.shape[-1]] = data
        mask[:, random_offset:random_offset+data.shape[-1]] = 1
        return canvas, mask
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.root, self.mode, self.data[index]))
        # load data from npz file
        z_audio=data["z_audio"]
        y_mask=data["y_mask"]
        z_text=data["z_text"]
        clap_embed=data["clap_embed"]

        # normalize
        z_audio = self.normalize_z_audio(torch.tensor(z_audio))
        z_text = self.normalize_z_text(torch.tensor(z_text))
        # scale to 0, 1
        z_audio = scale_0_1(z_audio)
        z_text = scale_0_1(z_text)

        # cut if necessary
        if z_audio.shape[-1] >= self.max_len_seq:
            z_audio = z_audio[:, :, :self.max_len_seq]
        if z_text.shape[-1] >= self.max_len_seq:
            z_text = z_text[:, :, :self.max_len_seq]
        if y_mask.shape[-1] >= self.max_len_seq:
            y_mask = y_mask[:, :, :self.max_len_seq]
        

        # pad and shift randomly
        max_lengths = max(z_audio.shape[-1], z_text.shape[-1], y_mask.shape[-1])
        if max_lengths >= self.max_len_seq:
            offset = 0
        else:
            offset = np.random.randint(0, self.max_len_seq - max_lengths)
        
        z_audio, z_audio_mask = self.pad_and_shift(z_audio, offset)
        z_text, z_text_mask = self.pad_and_shift(z_text, offset)
        y_mask, y_mask_mask = self.pad_and_shift(torch.from_numpy(y_mask), offset)

        # scale to -1, 1
        z_audio = z_audio * 2 - 1
        z_text = z_text * 2 - 1

        # convert to torch tensors and return
        data = dict(
            z_audio=z_audio.unsqueeze(0),
            z_audio_mask=z_audio_mask.unsqueeze(0).long(),
            z_text=z_text.unsqueeze(0),
            z_text_mask=z_text_mask.unsqueeze(0).long(),
            clap_embed=clap_embed
        )
        return data


