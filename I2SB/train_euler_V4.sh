#!/bin/bash

#SBATCH -n 6
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=V4
#SBATCH --output=V4.out
#SBATCH --error=V4.err
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20G
#SBATCH -A ls_drzrh

python train.py --ckpt V4 --n-gpu-per-node 4 --batch-size 64 --microbatch 4 --beta-max 1.0 --num-itr 50000 --model_conf_path configs/V4.yaml --log-writer wandb --ot-ode --add-x1-noise