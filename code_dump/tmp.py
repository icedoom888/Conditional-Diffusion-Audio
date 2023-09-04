import torch.multiprocessing as mp
import numpy as np
import os
from vits import utils_diffusion
from tqdm import tqdm

if __name__ == '__main__':

    model, hps = utils_diffusion.load_vits_model(hps_path="vits/configs/ljs_base.json", checkpoint_path="vits/pretrained_ljs.pth")
    model.cuda()
    model.eval()

    phoneme_embedder = utils_diffusion.get_phoneme_embedder(model)

    text_path = '/home/alberto/conditional-diffusion-audio/data/flickr_audio/texts/'
    text_files = [os.path.join(text_path, f) for f in os.listdir(text_path)]

    num_processes = 8
    all_sizes = []

    for text_file in tqdm(text_files):
        with open(text_file) as file:
            text = file.read()

        x, _, _, _ = phoneme_embedder(text, hps=hps)
        
        sizes = all_sizes.append(x.size()[2])

        print(f'Avg size: {np.mean(all_sizes)}')
        print(f'Max size: {np.max(all_sizes)}')
        print(f'Min size: {np.min(all_sizes)}')

    print(f'Avg size: {np.mean(all_sizes)}')
    print(f'Max size: {np.max(all_sizes)}')
    print(f'Min size: {np.min(all_sizes)}')

    # Flickr
    # Avg size: 117.77805
    # Max size: 7859
    # Min size: 1
