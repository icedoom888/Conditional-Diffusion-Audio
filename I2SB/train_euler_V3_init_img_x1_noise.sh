#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=v3_x1_noise_ii
#SBATCH --output=v3_x1_noise_ii.out
#SBATCH --error=v3_x1_noise_ii.err
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20G
#SBATCH -A ls_polle

python train.py --n-gpu-per-node 1 --batch-size 64 --microbatch 4 --beta-max 1.0 --num-itr 500000 --model_conf_path configs/V3_add_x1_noise_init_image.yaml --log-writer wandb --ot-ode --add-x1-noise