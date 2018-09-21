#!/bin/bash

python eval.py \
       --logtostderr \
       --checkpoint_dir=/media/panasonic/644E9C944E9C611A/tmp/model/20180910_ssd_mobilenet_v1_gyoza \
       --eval_dir=/media/panasonic/644E9C944E9C611A/tmp/model/20180910_ssd_mobilenet_v1_gyoza_validation  \
       --pipeline_config=ssd_mobilenet_v1_coco.config
