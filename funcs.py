from typing import Optional, Tuple, Dict
import torch.nn.functional as F
import torch
from torch import Tensor


def crop_or_pad_tensor(input_tensor, target_size):
    """
    Crop or pad the third dimension of the input tensor to the target size.

    Args:
    input_tensor (torch.Tensor): The input tensor to be cropped or padded (3D tensor).
    target_size (int): The target size of the third dimension.

    Returns:
    torch.Tensor: The cropped or padded tensor.
    """

    # Get the current size of the input tensor
    input_size = input_tensor.size(2)

    # Calculate the difference between the current size and the target size
    size_diff = target_size - input_size

    # Calculate the padding values
    padding = [0, size_diff]

    # Use F.pad to pad or crop the input tensor along the third dimension
    output_tensor = F.pad(input_tensor, (padding[0], padding[1], 0, 0, 0, 0))

    return output_tensor

def print_sizes(batch: Dict[str, Tensor]) -> None:
    '''Prints all sizes of tensor in a batch'''
    for key in batch.keys():
        print(f'{key} : {batch[key].size()}')
