#!/bin/bash

_MOV_DIR=$1
_TARGET_DIR=$2

echo ${_MOV_DIR}
echo ${_TARGET_DIR}
mkdir ${_TARGET_DIR}

count=0
for f in `ls ${_MOV_DIR} | grep mp4`
do
    echo $count
    echo $f
    dir_name=$(echo $count | xargs -P1 printf '%06d\n')
    echo $dir_name
    mkdir ${_TARGET_DIR}'/'$dir_name

    cd ${_TARGET_DIR}'/'$dir_name
    ffmpeg -i ${_MOV_DIR}'/'$f -s 512x512 %06d.png
    
    count=$((count+1))
done
