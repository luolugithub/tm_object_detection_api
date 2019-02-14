import os
import copy
import math
import time
import argparse

from PIL import Image, ImageDraw
import numpy as np

import xml.etree.ElementTree as ET


def visualize_annotation(image_file, annotation_dir):
    # image_file = os.path.join(_IMAGE_DIR, image_file)
    xml_file = os.path.basename(image_file).replace('.png', '.xml')
    xml_file = os.path.join(annotation_dir, xml_file)

    print(image_file)
    print(xml_file)

    tree = ET.ElementTree(file=xml_file)
    root = tree.getroot()

    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)

    for items  in root.findall('.object/bndbox'):
        print('loop1')
        start_pos = []
        end_pos = []
        pos = []
        for idx, item in enumerate(items):
            print('loop2')
            if str(item.tag) == 'xmin':
                start_pos.insert(0, item.text)
            elif str(item.tag) == 'ymin':
                start_pos.append(item.text)
            elif str(item.tag) == 'xmax':
                end_pos.insert(0, item.text)
            elif str(item.tag) == 'ymax':
                end_pos.append(item.text)
            if idx == (len(items) - 1):
                pos.extend(tuple(start_pos))
                pos.extend(tuple(end_pos))
                pos = tuple(pos)
                print(pos)
                draw.rectangle(
                    ((int(pos[0]), int(pos[1])), (int(pos[2]), int(pos[3]))),
                    outline=(255,0,0))
                img = img.resize((100,100))

    return np.asarray(img)


def main(image_dir, annotation_dir, target_dir):
    # generate target directory
    if os.path.isdir(target_dir) is False:
        os.mkdir(target_dir)

    # count number of files
    number_of_files = 0
    sub_directories = os.listdir(image_dir)
    for sub_directory in sub_directories:
        sub_directory = os.path.join(image_dir, sub_directory)
        number_of_files += len(os.listdir(sub_directory))

    # generatge array which includes all file path
    image_list = []
    for root, dirs, files in sorted(os.walk(image_dir)):
        for f in files:
            filepath = os.path.join(root, f)
            image_list.append(filepath)

    print('image_list', len(image_list))

    # # define progress bar
    # count = 0
    # number_of_files = len(image_list)
    # progress = "\r >> {0}/{1}"
    
    # # resize all image
    # resize_images = np.array([])
    # t0 = time.time()
    # image_array = np.array(
    #     [np.array(Image.open(i).resize((100,100))) for i in image_list]
    # )
    
    # image_array = np.array(
    #     [np.array(lambda x: visualize_annotation(i)) for i in image_list]
    # )

    image_array = []
    for f in image_list:
        annotation_img = visualize_annotation(f, annotation_dir)
        image_array.append(annotation_img)

    image_array = np.array(image_array)
    print(image_array)
    
    
    # # for idx, i in enumerate(range(len(image_list))):
    # for idx, i in enumerate(range(10)):
    #     img = Image.open(image_list[i]).resize((100,100))
    #     img = np.asarray(img)
    #     print(img)
    #     np.append(image_array, np.array(img))
    #     print(progress.format(idx, number_of_files),end='')
    #     count += 1
    # t1 = time.time()
    # print('\nresized time: ', t1 - t0)

    print('image_array.shape', image_array.shape)
    
    max_frames = int(np.ceil(np.sqrt(image_array.shape[0])))
    frames = []
    
    for i in range(image_array.shape[0]):
        try:
            f = Image.fromarray((image_array[i]))
            frames.append(f.getdata())
        except:
            print(f + ' is not a valid image')

    tile_width = frames[0].size[0]
    tile_height = frames[0].size[1]


    if len(frames) > max_frames:
        spritesheet_width = tile_width * max_frames
        spritesheet_height = tile_height * max_frames
    else:
        spritesheet_width = tile_width * len(frames)
        spritesheet_height = tile_height

    print('spritesheet_height', spritesheet_height)
    print('spritesheet_width', spritesheet_width)

    spritesheet = Image.new(
        'RGB',
        (int(spritesheet_width), int(spritesheet_height))
    )

    # define progress bar
    count = 0
    number_of_files = len(frames)
    progress = "\r >> {0}/{1}"

    for idx, f in enumerate(frames):
        top = tile_height * math.floor((frames.index(f)) / max_frames)
        left = tile_width * (frames.index(f) % max_frames)
        bottom = top + tile_height
        right = left + tile_width

        box = (left, top, right, bottom)
        box = [int(i) for i in box]
        cut_frame = f.crop((0,0,tile_width,tile_height))

        spritesheet.paste(cut_frame, box)
        print(progress.format(idx, number_of_files),end='')
        count += 1

    spritesheet.save(os.path.join(target_dir, 'splite.png'))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize annotation')
    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        type=str,
        default=None,
        help='please enter image file path'
    )
    parser.add_argument(
        '--annotation_dir',
        dest='annotation_dir',
        type=str,
        default=None,
        help='please enter xml file path',
    )
    parser.add_argument(
        '--target_dir',
        dest='target_dir',
        type=str,
        default=None,
        help='please enter filepath which want to output file',
    )
    argv = parser.parse_args()
        
    main(argv.image_dir, argv.annotation_dir, argv.target_dir)
