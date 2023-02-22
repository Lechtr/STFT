#!/bin/sh
source activate joshua_STFT
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/STFT/sun.yaml \
    OUTPUT_DIR log_dir/sun \
    # SOLVER.IMS_PER_BATCH 1 \
    # TEST.IMS_PER_BATCH 1