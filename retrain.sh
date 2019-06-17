#!/bin/bash
DATASET_DIR=$PWD/data/train
OUTPUT_GRAPH=$PWD/models/retrained/retrained_graph.pb
OUTPUT_LABELS=$PWD/models/retrained/retrained_labels.txt
TFHUB_MODULE=https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/3
BOTTLENECK_DIR=$PWD/data/bottleneck
SAVED_MODEL_DIR=$PWD/models/retrained/

TRAINING_STEPS=300000

python scripts/retrain.py \
    --how_many_training_steps=${TRAINING_STEPS} \
    --eval_step_interval=1000 \
    --output_graph=${OUTPUT_GRAPH} \
    --output_labels=${OUTPUT_LABELS} \
    --image_dir=${DATASET_DIR} \
    --tfhub_module=$TFHUB_MODULE \
    --bottleneck_dir=$BOTTLENECK_DIR \
    --flip_left_right \
    --random_scale=30 \
    --random_brightness=30 \
    --testing_percentage=10 \
    --validation_percentage=10 \
    --validation_batch_size=-1

OPTIMIZED_GRAPH=$PWD/models/retrained/optimized.pb

python scripts/optimize \
    --input=$OUTPUT_GRAPH \
    --output=$OPTIMIZED_GRAPH \
    --input_names=Placeholder \
    --output_names=final_result \
    --frozen_graph=true \
    --toco_compatible=true