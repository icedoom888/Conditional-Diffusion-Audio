STEPS=5

CKPT="V4"
CONF=$CKPT
python sample.py --ckpt $CKPT --batch-size 4 --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 --ot-ode --add-x1-noise --txt_embeds
python sample.py --ckpt $CKPT --batch-size 4 --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 --ot-ode --add-x1-noise --txt_embeds
python sample.py --ckpt $CKPT --batch-size 4 --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 --ot-ode --add-x1-noise --txt_embeds
python sample.py --ckpt $CKPT --batch-size 4 --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0 --ot-ode --add-x1-noise --txt_embeds