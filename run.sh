#!/bin/bash

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Testing model trained from scratch
echo -e "${GREEN}Testing model trained from scratch${NC}"
python train.py --mode "train" --test-only \
    --weights "weights/scratch_effnetv2m_mish_cos_e100.ckpt" --batch-size 64

echo

# Testing model fine-tuned
echo -e "${RED}Testing model fine-tuned${NC}"
python train.py --mode "finetune" --test-only \
    --weights "weights/tl_effnetv2m_in21k_e50.ckpt" --batch-size 64
