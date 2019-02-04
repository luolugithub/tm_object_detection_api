#!/bin/bash

INPUT_TYPE=image_tensor
PIPLINE_CONFIG_PATH=ssd_mobilenet_v1_coco.config
TRAINED_CKPT_PREFIX=/media/tsukudamayo/0CE698DCE698C6FC/tmp/model/20190204_ssd_mobilenet_v1_gyoza_cooking/model.ckpt-21247
EXPORT_DIR=export/export_20190205_21247

python export_inference_graph.py \
       --input_type=${INPUT_TYPE} \
       --pipeline_config_path=${PIPLINE_CONFIG_PATH} \
       --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
       --output_directory=${EXPORT_DIR}
