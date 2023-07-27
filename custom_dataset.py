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


def find_1d_bounding_box(x):
    starts = torch.argmax(x, dim=-1, keepdim=False)
    ends = torch.tensor([x.shape[-1]]*starts.shape[0]) - torch.argmax(x.flip((-1,)), dim=-1, keepdim=False)
    return starts, ends


def scale_0_1(x, eps=1e-7, return_max_min=False):
    x_max, x_min = x.max(), x.min()
    x_scaled = (x - x_min) / (x_max - x_min + eps)
    if return_max_min:
        return x_scaled, (x_max, x_min)
    return x_scaled


def min_max_normalize(x, min, max):
    x = (x - min) / (max - min)
    return x


def min_max_denormalize(x, min, max):
    x = x * (max - min) + min
    return x


def scale_01_to_11(x):
    normalize = Normalize(mean=[0.5], std=[0.5])
    return normalize(x)


def scale_to_minus11(x):
    x = scale_0_1(x)
    return x * 2 - 1


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
    def __init__(self, root, mode="train", max_len_seq=384, normalization=False):
        super().__init__()
        self.root = root
        self.mode = mode
        self.max_len_seq = max_len_seq
        self.load_data()
        self.normalize = normalization
        self.normalize_z_audio = Normalize(mean=[Z_AUDIO_MEAN], std=[Z_AUDIO_STD])
        self.normalize_z_text = Normalize(mean=[Z_TEXT_MEAN], std=[Z_TEXT_STD])

    def load_data(self):
        self.data = os.listdir(os.path.join(self.root, self.mode))
        assert len(self.data) > 0, "No data found"
    
    def zero_pad_and_shift(self, data, random_offset=None):
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
        z_audio=torch.from_numpy(data["z_audio"])
        y_mask=torch.from_numpy(data["y_mask"])
        z_text=torch.from_numpy(data["z_text"])
        clap_embed=torch.from_numpy(data["clap_embed"])

        # cut if necessary
        if z_audio.shape[-1] >= self.max_len_seq:
            z_audio = z_audio[:, :, :self.max_len_seq]
        if z_text.shape[-1] >= self.max_len_seq:
            z_text = z_text[:, :, :self.max_len_seq]
        if y_mask.shape[-1] >= self.max_len_seq:
            y_mask = y_mask[:, :, :self.max_len_seq]
        

        # pad and shift randomly
        z_audio_length, z_text_length, y_mask_length = z_audio.shape[-1], z_text.shape[-1], y_mask.shape[-1]
        max_lengths = max(z_audio_length, z_text_length, y_mask_length)
        if max_lengths >= self.max_len_seq:
            offset = 0
        else:
            offset = np.random.randint(0, self.max_len_seq - max_lengths)
        
        z_audio, z_audio_mask = self.zero_pad_and_shift(z_audio, offset)
        z_text, z_text_mask = self.zero_pad_and_shift(z_text, offset)
        y_mask, y_mask_mask = self.zero_pad_and_shift(y_mask, offset)

        # convert to torch tensors and return
        data = dict(
            z_audio=z_audio.unsqueeze(0),
            z_audio_mask=z_audio_mask.unsqueeze(0).long(),
            z_text=z_text.unsqueeze(0),
            z_text_mask=z_text_mask.unsqueeze(0).long(),
            offset=offset,
            z_audio_length=z_audio_length,
            y_mask=y_mask,
            clap_embed=clap_embed
        )
        return data


