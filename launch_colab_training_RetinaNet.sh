#!/bin/sh
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/train_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/MEGA/JF_KUMC_RetinaNet.yaml \
    OUTPUT_DIR log_dir/KUMC_RetinaNet \
  #   MODEL.WEIGHT log_dir/KUMC/last_checkpoint \
    SOLVER.IMS_PER_BATCH 1 \
    TEST.IMS_PER_BATCH 1