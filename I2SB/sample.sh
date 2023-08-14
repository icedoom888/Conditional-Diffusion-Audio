STEPS=20

CKPT="V2"
CONF=$CKPT
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 --ot-ode 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0 --ot-ode

CKPT="V2_init_image"
CONF=$CKPT
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 --ot-ode 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 --ot-ode 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0 --ot-ode

CKPT="V2_add_x1_noise"
CONF=$CKPT
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 --ot-ode
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0 --ot-ode

CKPT="V2_no_ote"
CONF=$CKPT
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0