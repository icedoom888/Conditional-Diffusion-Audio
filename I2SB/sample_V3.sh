STEPS=20

CKPT="V3_add_x1_noise_init_image"
CONF=$CKPT
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 --ot-ode 
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0 --ot-ode

CKPT="V3_add_x1_noise"
CONF=$CKPT
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 --ot-ode 
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 --ot-ode 
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0 --ot-ode

CKPT="V3_init_image"
CONF=$CKPT
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1  --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0 --ot-ode