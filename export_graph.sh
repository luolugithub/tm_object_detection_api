#!/bin/bash

INPUT_TYPE=image_tensor
PIPLINE_CONFIG_PATH=ssd_mobilenet_v1_coco.config
TRAINED_CKPT_PREFIX=/media/panasonic/644E9C944E9C611A/tmp/model/20180913_ssd_mobilenet_v1_gyoza_cookware/model.ckpt-200000
EXPORT_DIR=export/export_20180914

python export_inference_graph.py \
       --input_type=${INPUT_TYPE} \
       --pipeline_config_path=${PIPLINE_CONFIG_PATH} \
       --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
       --output_directory=${EXPORT_DIR}
