#!/bin/bash


python create_ingradient_tf_record.py \
       --data_dir=/media/panasonic/644E9C944E9C611A/tmp/data/detection/20181001_gyoza \
       --output_path=/media/panasonic/644E9C944E9C611A/tmp/data/detection/20181001_gyoza/record/cooking_train.record \
       --label_map_path=object_detection/data/cooking_20181001_label_map.pbtxt 

