#!/bin/bash
INPUT_GRAPH=${PWD}/models/retrained/retrained_graph.pb
OUTPUT_GRAPH=${PWD}/models/model.tflite
OUTPUT_NODE_NAMES=final_result
TYPE=FLOAT

tflite_convert \
    --output_file=$OUTPUT_GRAPH \
    --graph_def_file=$INPUT_GRAPH \
    --input_arrays=Placeholder \
    --output_arrays=$OUTPUT_NODE_NAMES \
    --input_format=TENSORFLOW_GRAPHDEF \
    --output_format=TFLITE \
    --input_shape=1,224,224,3 \
    --inference_type=$TYPE \
    --inference_input_type=$TYPE

cp $PWD/models/retrained/retrained_labels.txt $PWD/models/labels.txt