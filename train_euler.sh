#!/bin/bash

#SBATCH -n 2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=V4
#SBATCH --output=V4.out
#SBATCH --error=V4.err
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20G
#SBATCH -A ls_drzrh

python train.py --config configs/V4.yaml