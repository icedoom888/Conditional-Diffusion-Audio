#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=v3
#SBATCH --output=v3.out
#SBATCH --error=v3.err
#SBATCH --gpus=1
#SBATCH --gres=gpumem:50G
#SBATCH -A ls_drzrh

python train.py --config configs/V3.yaml