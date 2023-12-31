import sys, os
dirname = os.path.dirname(__file__)
sys.path.append(dirname)

from mel_processing import spectrogram_torch
from models import SynthesizerTrn
from text.symbols import symbols
import vits.utils as vits_utils
import torch
from text import text_to_sequence
import commons
from transformers import AutoTokenizer, ClapTextModelWithProjection, ClapConfig, ClapModel, AutoFeatureExtractor, AutoModel
import torch.nn.functional as F

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

    def Z_to_audio(z, y_mask, sid=None, grad=False):
        if sid is not None:
            g = net_g.emb_g(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None
        
        # decode latent space to audio
        if grad:
            o_hat = net_g.dec(z * y_mask, g=g)
        else:
            with torch.no_grad():
                o_hat = net_g.dec(z * y_mask, g=g)
        return o_hat
    return Z_to_audio


def mp_to_zp(m_p, logs_p, noise_scale=.667):
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    return z_p


def get_Z_preflow_to_audio(net_g):
    """Returns a function that takes latent space representation and returns audio.
    
    Args:
        net_g (SynthesizerTrn): The VITS model.

    Returns:
        Z_to_audio (function): A function that takes latent space representation and returns audio.  
    """

    def func(z_p, y_mask, sid=None):
        if sid is not None:
            g = net_g.emb_g(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None
        
        # decode latent space to audio
        with torch.no_grad():
            z = net_g.flow(z_p, y_mask, g=g, reverse=True)
            o_hat = net_g.dec(z * y_mask, g=g)
        return o_hat
    return func


def get_text_to_Z(net_g):
    """Returns a function that takes text and returns the latent space representation.

    Args:
        text (str): The text to be converted to latent space representation.
        net_g (SynthesizerTrn): The VITS model.

    Returns:
        text_to_Z (function): A function that takes text and returns the latent space representation.
    """

    def text_to_Z(text, hps, sid=None, noise_scale=.667, length_scale=1, noise_scale_w=0.8, max_len=None, y_lengths=None):
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            net_out = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w, max_len=max_len, y_lengths=y_lengths)
            z = net_out[3][0]
            y_mask = net_out[2]
        return z, y_mask
    return text_to_Z

def get_phoneme_embedder(net_g):

    def text_to_phoneme_emb(text, hps):
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            # sid = torch.LongTensor([1]).cuda() speaker ID
            x, m, logs, x_mask = net_g.enc_p(x_tst, x_tst_lengths)
        return x, m, logs, x_mask
    
    return text_to_phoneme_emb


def get_text_to_Z_preflow(net_g):
    """Returns a function that takes text and returns the latent space representation.

    Args:
        text (str): The text to be converted to latent space representation.
        net_g (SynthesizerTrn): The VITS model.

    Returns:
        text_to_Z (function): A function that takes text and returns the latent space representation.
    """

    def text_to_Z_preflow(text, hps, sid=None, noise_scale=.667, length_scale=1, noise_scale_w=0.8, max_len=None, y_lengths=None):
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            net_out = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w, max_len=max_len, y_lengths=y_lengths)
            m_p, logs_p = net_out[3][2], net_out[3][3]
            y_mask = net_out[2]
        return m_p, logs_p, y_mask
    return text_to_Z_preflow


def get_text_embedder(model="CLAP"):
    if model == "CLAP":
        model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused").to("cuda", non_blocking=True)
        tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused").to("cuda", non_blocking=True)

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

        def audio_embedder(audio):
            
            # upsample the audio to a sampling rate of 48000 before passing it to the audio embedder
            inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=48000)

            for k, v in inputs.items():
                inputs[k] = v.to("cuda", non_blocking=True)

            audio_features = model.get_audio_features(**inputs)
            return audio_features
        return audio_embedder


def get_sentence_embedder(model='all-MiniLM-L6-v2'):
    if model == 'all-MiniLM-L6-v2':
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        def sentence_embedder(sentence):
            encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            return sentence_embeddings

        return sentence_embedder
 
    else:
        raise NotImplementedError
    


