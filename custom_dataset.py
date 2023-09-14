from torch.utils.data import Dataset
import torch
import os
import numpy as np
from torchvision.transforms import Normalize
from funcs import print_sizes, crop_or_pad_tensor

# stats for normalization
Z_AUDIO_MEAN_SMALL = -0.0270
Z_AUDIO_STD_SMALL = 1.0935
Z_TEXT_MEAN_SMALL = -0.0277
Z_TEXT_STD_SMALL = 1.1185

Z_TEXT_MEAN = -0.0410
Z_TEXT_STD = 1.3542

Z_AUDIO_MEAN = -0.0405
Z_AUDIO_STD = 1.3229


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


class SlidingWindow(Dataset):
    """
    Dataset  with sliding window

    Args:
        root (str): path to the root of the dataset
        mode (str): train or test
        max_len_seq (int): maximum length of the sequence
        z_downsampling_factor (int): downsampling factor of the z vectors of the vits encoder
        sr (int): sampling rate of the audio
        mean_on (str): "audio" or "text"
        normalize (bool): normalize the z vectors
    Returns:
        dict: dictionary of the data
    """
    def __init__(self, root, mode="train", max_len_seq=384, z_downsampling_factor=8, sr=22050, mean_on="audio", normalize=True):
        super().__init__()
        self.root = root
        self.mode = mode
        self.sr = sr
        self.z_to_audio = 2 ** z_downsampling_factor
        self.max_len_seq = max_len_seq
        self.normalization = normalize

        # normalization initialization
        assert mean_on in ["audio", "text"]
        self.z_normalize = Normalize(
            mean=[Z_TEXT_MEAN if mean_on == "text" else Z_AUDIO_MEAN],
            std=[Z_TEXT_STD if mean_on == "text" else Z_AUDIO_STD]
            )
        self.load_data()
        print(f"Created {self.mode} dataset with {len(self)} samples")
    
    def normalize(self, x):
        x = self.z_normalize(x)
        x = scale_to_minus11(x)
        return x
    
    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        self.data = os.listdir(os.path.join(self.root, self.mode))
        assert len(self.data) > 0, "No data found"

    def zero_pad(self, data):
        canvas = torch.zeros((1, data.shape[-2], self.max_len_seq))
        mask = canvas.clone()
        canvas[..., :data.shape[-1]] = data
        mask[..., :data.shape[-1]] = 1
        return canvas, mask
    

    def __getitem__(self, index):
        data = np.load(os.path.join(self.root, self.mode, self.data[index]))

        z_audio = torch.from_numpy(data["z_audio"])
        y_mask = torch.from_numpy(data["y_mask"])
        z_text = torch.from_numpy(data["z_text"])
        clap_embed = torch.from_numpy(data["clap_embed"])
        sentence_embed = torch.from_numpy(data["sentence_embed"])
        phoneme_embed = torch.from_numpy(data["phoneme_embed"])

        audio = torch.from_numpy(data["audio"])

        audio_lenght = audio.size()[0]

        if audio.ndim == 1:
            audio = audio[None, :]
        elif audio.shape[-2] == 2:
            audio = audio.mean(-2, keepdim=True)
        

        if z_audio.shape[-1] > self.max_len_seq and z_text.shape[-1] > self.max_len_seq:

            seq_len = min(z_audio.shape[-1], z_text.shape[-1])

            # take random slize
            random_offset = torch.randint(0, seq_len - self.max_len_seq, (1,)).item()

            # take slices
            z_audio = z_audio[..., random_offset:random_offset+self.max_len_seq]
            y_mask = y_mask[..., random_offset:random_offset+self.max_len_seq]
            z_text = z_text[..., random_offset:random_offset+self.max_len_seq]
            audio = audio[..., random_offset*self.z_to_audio:(random_offset+self.max_len_seq)*self.z_to_audio]

            # make dummy masks
            z_audio_mask = torch.ones_like(z_audio)
            z_text_mask = torch.ones_like(z_text)
            y_mask_mask = torch.ones_like(y_mask)
        
        elif z_audio.shape[-1] > self.max_len_seq and z_text.shape[-1] < self.max_len_seq:

            seq_len = z_audio.shape[-1]

            # take random slize
            random_offset = torch.randint(0, seq_len - self.max_len_seq, (1,)).item()

            # take slices
            z_audio = z_audio[..., random_offset:random_offset+self.max_len_seq]
            z_text, z_text_mask = self.zero_pad(z_text)
            y_mask, y_mask_mask = self.zero_pad(y_mask)
            audio = audio[..., random_offset*self.z_to_audio:(random_offset+self.max_len_seq)*self.z_to_audio]

            # make dummy masks
            z_audio_mask = torch.ones_like(z_audio)

        elif z_audio.shape[-1] < self.max_len_seq and z_text.shape[-1] > self.max_len_seq:

            seq_len = z_text.shape[-1]

            # take random slize
            random_offset = torch.randint(0, seq_len - self.max_len_seq, (1,)).item()

            # take slices
            z_audio, z_audio_mask = self.zero_pad(z_audio)
            y_mask = y_mask[..., random_offset:random_offset+self.max_len_seq]
            z_text = z_text[..., random_offset:random_offset+self.max_len_seq]
            audio = torch.cat([audio, torch.zeros((1, max(0, self.max_len_seq*self.z_to_audio - audio.shape[-1])))], dim=-1)

            # make dummy masks
            z_text_mask = torch.ones_like(z_text)
            y_mask_mask = torch.ones_like(y_mask)

        else:
            # pad it
            z_audio, z_audio_mask = self.zero_pad(z_audio)
            z_text, z_text_mask = self.zero_pad(z_text)
            y_mask, y_mask_mask = self.zero_pad(y_mask)
            audio = torch.cat([audio, torch.zeros((1, max(0, self.max_len_seq*self.z_to_audio - audio.shape[-1])))], dim=-1)

        # make sure audio is perfect length
        audio = audio[..., :self.max_len_seq*self.z_to_audio]
        
        if self.normalization:
            z_audio = self.normalize(z_audio)
            z_text = self.normalize(z_text)
            audio = audio / max(audio.max(), -audio.min())
        
        phoneme_embed = crop_or_pad_tensor(phoneme_embed, target_size=256)
        
        data = {
            "z_audio": torch.squeeze(z_audio, 0),
            "y_mask": torch.squeeze(y_mask, 0),
            "z_text": torch.squeeze(z_text, 0),
            "clap_embed": clap_embed,
            "sentence_embed": sentence_embed,
            "phoneme_embed": phoneme_embed,
            "audio": audio,
            "audio_lenght": audio_lenght,
            "z_audio_mask": torch.squeeze(z_audio_mask, 0),
            "z_text_mask": torch.squeeze(z_text_mask, 0),
            "y_mask_mask": torch.squeeze(y_mask_mask, 0),
            "offset": 0,
            "z_audio_length": self.max_len_seq,
        }

        return data
 
        
class Latent(Dataset):
    def __init__(self, root, mode="train", max_len_seq=384, normalization=False):
        super().__init__()
        self.root = root
        self.mode = mode
        self.max_len_seq = max_len_seq
        self.load_data()
        self.normalize = normalization

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


class Latent_Audio(Dataset):
    def __init__(self, root, mode="train", max_len_seq=384, normalization=False):
        super().__init__()
        self.root = root
        self.mode = mode
        self.max_len_seq = max_len_seq
        self.load_data()
        self.normalize = normalization
        self.normalize_z_audio = Normalize(mean=[Z_AUDIO_MEAN], std=[Z_AUDIO_STD])
        self.normalize_z_text = Normalize(mean=[Z_TEXT_MEAN], std=[Z_TEXT_STD])
        self.hop_length = 256
        self.mean_pitch = 137.61
        self.std_pitch = 49.64

    def load_data(self):
        self.data = os.listdir(os.path.join(self.root, self.mode))
        assert len(self.data) > 0, "No data found"
    
    def zero_pad_and_shift(self, data, random_offset=None):
        canvas = torch.zeros((data.shape[-2], self.max_len_seq))
        mask = canvas.clone()
        canvas[:, random_offset:random_offset+data.shape[-1]] = data
        mask[:, random_offset:random_offset+data.shape[-1]] = 1
        return canvas, mask
    
    def zero_pad_and_shift_audio(self, data, random_offset=None):
        canvas = torch.zeros(self.max_len_seq*self.hop_length)
        mask = canvas.clone()
        canvas[random_offset*self.hop_length:random_offset*self.hop_length+data.shape[0]] = data
        mask[random_offset*self.hop_length:random_offset*self.hop_length+data.shape[0]] = 1
        return canvas, mask
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.root, self.mode, self.data[index]))

        # load data from npz file
        audio=torch.from_numpy(data['audio'])
        z_audio=torch.from_numpy(data["z_audio"])
        y_mask=torch.from_numpy(data["y_mask"])
        z_text=torch.from_numpy(data["z_text"])
        clap_embed=torch.from_numpy(data["clap_embed"])

        # cut if necessary
        if int(np.floor(audio.shape[-1]/self.hop_length)) >= self.max_len_seq:
            audio = audio[:, :, :self.max_len_seq*self.hop_length]
        if z_audio.shape[-1] >= self.max_len_seq:
            z_audio = z_audio[:, :, :self.max_len_seq]
        if z_text.shape[-1] >= self.max_len_seq:
            z_text = z_text[:, :, :self.max_len_seq]
        if y_mask.shape[-1] >= self.max_len_seq:
            y_mask = y_mask[:, :, :self.max_len_seq]
        

        # pad and shift randomly
        audio_lenght, z_audio_length, z_text_length, y_mask_length = int(np.floor(audio.shape[-1]/self.hop_length)), z_audio.shape[-1], z_text.shape[-1], y_mask.shape[-1]
        max_lengths = max(audio_lenght, z_audio_length, z_text_length, y_mask_length)
        if max_lengths >= self.max_len_seq:
            offset = 0
        else:
            offset = np.random.randint(0, self.max_len_seq - max_lengths)

        audio, audio_mask = self.zero_pad_and_shift_audio(audio, offset)
        z_audio, z_audio_mask = self.zero_pad_and_shift(z_audio, offset)
        z_text, z_text_mask = self.zero_pad_and_shift(z_text, offset)
        y_mask, y_mask_mask = self.zero_pad_and_shift(y_mask, offset)
        # convert to torch tensors and return
        data = dict(
            audio=audio[None,:],
            z_audio=z_audio,
            z_audio_mask=z_audio_mask.long(),
            z_text=z_text,
            z_text_mask=z_text_mask.long(),
            offset=offset,
            z_audio_length=z_audio_length,
            y_mask=y_mask,
            clap_embed=clap_embed
        )
        return data

if __name__ == "__main__":

    dataset = SlidingWindow(root='/home/alberto/conditional-diffusion-audio/data/processed_LJSpeech', mode="train")

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    for step, batch in enumerate(train_dataloader):
        print_sizes(batch)
        exit()