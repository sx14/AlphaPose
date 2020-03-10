set -x

# CONFIG=$1
# CKPT=$2
# VIDEO=$3

CONFIG=configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml
CKPT=pretrained_models/fast_421_res152_256x192.pth
VIDEO=/media/sunx/Data/2793806282.mp4

OUTDIR=${4:-"./examples/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
