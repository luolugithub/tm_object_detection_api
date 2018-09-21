#!/bin/bash


python create_ingradient_tf_record.py \
       --data_dir=/media/panasonic/644E9C944E9C611A/tmp/data/detection/20180912_gyoza_cookware/train \
       --output_path=/media/panasonic/644E9C944E9C611A/tmp/data/detection/20180912_gyoza_cookware/train/record/cooking_train.record \
       --label_map_path=object_detection/data/cooking_20180912_label_map.pbtxt 

