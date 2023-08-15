import sys, os
dirname = os.path.dirname(__file__)
sys.path.append(dirname)

from mel_processing import spectrogram_torch
from models import SynthesizerTrn
from text.symbols import symbols
import utils as vits_utils
import torch
from text import text_to_sequence
import commons
from transformers import AutoTokenizer, ClapTextModelWithProjection, ClapModel, AutoFeatureExtractor

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def load_vits_model(hps_path="./configs/ljs_base.json", device="cuda", checkpoint_path="pretrained_ljs.pth"):
    """Loads the VITS model from a checkpoint and returns it.
    
    Args:
        hps_path (str, optional): Path to the hparams file. Defaults to "./configs/ljs_base.json".
        device (str, optional): Device to load the model on. Defaults to "cuda".
        checkpoint_path (str, optional): Path to the checkpoint file. Defaults to "pretrained_ljs.pth".
    
    Returns:
        net_g (SynthesizerTrn): The VITS model.
        hps (dict): The hparams dictionary.
    """

    hps = vits_utils.get_hparams_from_file(hps_path)
    # initialize the vits model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    
    net_g.eval()
    vits_utils.load_checkpoint(checkpoint_path, net_g, None)
    return net_g, hps


def get_audio_to_Z(net_g, hps):
    """Returns a function that takes audio and returns the latent space representation.

    Args:
        net_g (SynthesizerTrn): The VITS model.
        hps (dict): The hparams dictionary.

    Returns:
        audio_to_Z (function): A function that takes audio and returns the latent space representation.
    """

    def audio_to_Z(audio, sid=None):
        assert audio.ndim == 2, "Audio must be tensor of shape C X T"
        spec = spectrogram_torch(
            y = audio,
            n_fft = hps.data.filter_length,
            sampling_rate=hps.data.sampling_rate,
            hop_size = hps.data.hop_length,
            win_size=hps.data.win_length,
        )

        # move to GPU
        y, y_lengths = spec.cuda(), torch.tensor([spec.shape[-1]]).cuda()

        if sid is not None:
            g = net_g.emb_g(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None

        with torch.no_grad():
            # encode audio to latent space (no identity used)
            z, m_q, logs_q, y_mask = net_g.enc_q(y, y_lengths, g=g)

        data = dict(
            z=z,
            m_q=m_q,
            logs_q=logs_q,
            y_mask=y_mask,
        )
        return data
    return audio_to_Z


def get_Z_to_audio(net_g):
    """Returns a function that takes latent space representation and returns audio.
    
    Args:
        net_g (SynthesizerTrn): The VITS model.

    Returns:
        Z_to_audio (function): A function that takes latent space representation and returns audio.  
    """

    def Z_to_audio(z, y_mask, sid=None):
        if sid is not None:
            g = net_g.emb_g(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None
        
        # decode latent space to audio
        with torch.no_grad():
            o_hat = net_g.dec(z * y_mask, g=g)
        return o_hat
    return Z_to_audio


def get_text_to_Z(net_g):
    """Returns a function that takes text and returns the latent space representation.

    Args:
        text (str): The text to be converted to latent space representation.
        net_g (SynthesizerTrn): The VITS model.

    Returns:
        text_to_Z (function): A function that takes text and returns the latent space representation.
    """

    def text_to_Z(text, hps, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None, y_lengths=None):
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            net_out = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w, max_len=max_len, y_lengths=y_lengths)
            z = net_out[3][0]
            y_mask = net_out[2]
        return z, y_mask
    return text_to_Z


def get_text_embedder(model="CLAP"):
    if model == "CLAP":
        model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        def text_embedder(text):
            tokens = tokenizer(text, padding=True, return_tensors="pt")
            embeds = model(**tokens)['text_embeds']
            return embeds
        return text_embedder
    else:
        raise NotImplementedError


def get_audio_embedder(model="CLAP"):
    if model == "CLAP":
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda", non_blocking=True)
        feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        print("Make sure you upsample the audio to a sampling rate of 48000 before passing it to the audio embedder!")

        def audio_embedder(audio):
            inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=48000)

            for k, v in inputs.items():
                inputs[k] = v.to("cuda", non_blocking=True)

            audio_features = model.get_audio_features(**inputs)
            return audio_features
        return audio_embedder
