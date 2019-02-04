#!/bin/bash


python create_ingradient_tf_record.py \
       --data_dir=/media/tsukudamayo/0CE698DCE698C6FC/tmp/data/dataset/gyoza_20190203_00-04/annotation/annotation \
       --output_path=/media/tsukudamayo/0CE698DCE698C6FC/tmp/data/tfrecord/20190203_00-04/record/cooking_train.record \
       --label_map_path=object_detection/data/gyoza_20190203_00-04_label_map.pbtxt

