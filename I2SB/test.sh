python train.py --name test --n-gpu-per-node 1 --batch-size 64 --microbatch 2 --ot-ode --beta-max 1.0 --model_conf_path configs/V4.yaml --add-x1-noise --log-writer wandb