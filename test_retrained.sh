#!/bin/bash
IMAGE_PATH=$PWD/data/test/
RETRAINED_GRAPH=$PWD/models/retrained/optimized.pb
RETRAINED_LABELS=$PWD/models/retrained/retrained_labels.txt
SIZE=224


python scripts/test_retrained.py \
    --graph=$RETRAINED_GRAPH --labels=$RETRAINED_LABELS \
    --input_layer=Placeholder \
    --input_height=$SIZE \
    --input_width=$SIZE \
    --output_layer=final_result \
    --image=$IMAGE_PATH