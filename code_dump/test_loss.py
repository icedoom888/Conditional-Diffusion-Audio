# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py
import argparse
import torch
from accelerate.logging import get_logger
from diffusers.training_utils import set_seed
from omegaconf import OmegaConf
from utils import CompositeLoss, get_datasets, get_model
from tqdm import tqdm
from funcs import print_sizes
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def main(conf):
    train_args = conf.training
    model_args = conf.model
    loss_args = conf.loss

    # seed alls

    set_seed(42)
    # Get datasets and dataloaders
    train_dataset, val_dataset = get_datasets(train_args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.train_batch_size, shuffle=False)

    # setup diffusion loss
    loss_fn = CompositeLoss(loss_args, train_args)

    # Set up model
    model = get_model(model_args, train_args, loss_fn)

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

    # extract the batch
    for batch in tqdm(train_dataloader):

        audio = batch["audio"]

        # Turn noise into new audio sample with diffusion
        loss, loss_dict = loss_fn(audio, torch.rand_like(audio))
        print(loss_dict)

    return


if __name__ == "__main__":
    # parse arguments for rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_conf.yaml")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    main(conf)
