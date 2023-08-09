import torch.nn as nn
import torch
import auraloss
from typing import Optional, Tuple, Dict
from torch import Tensor

class CompositeLoss(nn.Module):
    def __init__(self, loss_args: dict):
        self.losses = []
        self.weights = []
        self.loss_names = []

        # add all losses
        if loss_args.use_l1 == True:
            self.losses.append(torch.nn.functional.l1_loss)
            self.weights.append(loss_args.l1_weight)
            self.loss_names.append('l1_loss')
        
        if loss_args.use_l2 == True:
            self.losses.append(torch.nn.functional.l2_loss)
            self.weights.append(loss_args.l2_weight)
            self.loss_names.append('l2_loss')
        
        if loss_args.use_mrstft == True:
            loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
                            fft_sizes=[1024, 2048, 8192],
                            hop_sizes=[256, 512, 2048],
                            win_lengths=[1024, 2048, 8192],
                            scale="mel",
                            n_bins=128,
                            sample_rate=22050,
                            perceptual_weighting=True,
                        )
            self.losses.append(loss_fn)
            self.weights.append(loss_args.mrstft_weight)
            self.loss_names.append('mrstft_loss')

        super(CompositeLoss, self).__init__()

    def forward(self, inputs: Tensor, targets) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss = 0
        loss_dict = {}

        for loss_fn, weight, loss_name in zip(self.losses, self.weights, self.loss_names):
            loss_item = loss_fn(inputs, targets)
            loss_dict[loss_name] = loss_item

            loss += weight * loss_item

        return loss, loss_dict


def print_sizes(batch: Dict[str, Tensor]) -> None:
    '''Prints all sizes of tensor in a batch'''
    for key in batch.keys():
        print(f'{key} : {batch[key].size()}')