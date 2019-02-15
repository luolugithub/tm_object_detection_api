import os
import copy
import math
import time

from PIL import Image
import numpy as np


_IMAGE_DIR = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/splite_sample/image'
_TARGET_DIR = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/splite_sample'


def main():
    # generate target directory
    if os.path.isdir(_TARGET_DIR) is False:
        os.mkdir(_TARGET_DIR)

    # count number of files
    number_of_files = 0
    sub_directories = os.listdir(_IMAGE_DIR)
    for sub_directory in sub_directories:
        sub_directory = os.path.join(_IMAGE_DIR, sub_directory)
        number_of_files += len(os.listdir(sub_directory))

    # generatge array which includes all file path
    image_list = []
    for root, dirs, files in sorted(os.walk(_IMAGE_DIR)):
        for f in files:
            filepath = os.path.join(root, f)
            image_list.append(filepath)

    print('image_list', len(image_list))

    # image_array = np.array([np.array(Image.open(i) for i in image_list)])

    # define progress bar
    count = 0
    number_of_files = len(image_list)
    progress = "\r >> {0}/{1}"
    
    # resize all image
    resize_images = np.array([])
    t0 = time.time()
    # image_array = np.array(
    #     [np.array(Image.open(i).resize((100,100))) for i in image_list]
    # )
    image_array = np.array(
        [np.array(Image.open(i)) for i in image_list]
    )
    
    # # for idx, i in enumerate(range(len(image_list))):
    # for idx, i in enumerate(range(10)):
    #     img = Image.open(image_list[i]).resize((100,100))
    #     img = np.asarray(img)
    #     print(img)
    #     np.append(image_array, np.array(img))
    #     print(progress.format(idx, number_of_files),end='')
    #     count += 1
    t1 = time.time()
    print('\nresized time: ', t1 - t0)

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

    spritesheet.save(os.path.join(_TARGET_DIR, 'splite.png'))

    
if __name__ == '__main__':
    main()
