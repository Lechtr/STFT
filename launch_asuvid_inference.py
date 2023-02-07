#!/bin/sh
source activate joshua_STFT
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/test_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/STFT/asuvid_R_50_STFT.yaml \
    MODEL.WEIGHT pretrained_models/ASUMayo_STFT_R_50.pth \
    OUTPUT_DIR log_dir/asuvid_R_50_STFT \
    TEST.IMS_PER_BATCH 4
