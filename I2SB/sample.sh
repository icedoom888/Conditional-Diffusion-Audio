CKPT="V2_init_image"
CONF=$CKPT
STEPS=20

python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 1.0 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 3.0 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 5.0 
python sample.py --ckpt $CKPT --n-gpu-per-node 1 --dataset-dir ../vits/LJSProcessedFull --model_conf_path configs/$CONF.yaml --nfe $STEPS --cfg 10.0