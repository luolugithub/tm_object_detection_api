import os
import shutil
import argparse

import xml.dom.minidom

import numpy as np
import cv2

# output_dir = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/ssd_food_dossari_20180903_cu_ep_tm_train_1000'
# original_dir = '/media/panasonic/644E9C944E9C611A/tmp/data/img/food_dossari_20180815_cu_ep_tm_1000'

#-----------------------------------------#
# image file data which want to parse xml #
#-----------------------------------------#
break_flag = 0
x1, y1, x2, y2 = 0, 0, 0, 0
filename = str()
name = str()
width, height, depth = int(), int(), int()
xmin, ymin, xmax, ymax = int(), int(), int(), int()


def callback(event, x, y, flags, params):
    """ recieve a mouse event """
    global x1, y1, x2, y2, xmin, ymin, xmax, ymax, img, break_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        x1 = x
        y1 = y
    elif event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        x2 = x
        y2 = y
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        break_flag = 1


def generate_xml(image_file, annotation_dir, parse_xml):
    """
    output xml file
    input:
      image_file: strings of image file
      annotation_dir: the directory that you want to output xml file 
      parse_xml: xml.dom.minidom.Document().toprettyxml()
    return:
      output xml files to annotation_dir
    """
    xml_filename, _ = os.path.splitext(image_file)
    output_xml = os.path.join(annotation_dir, xml_filename + '.xml')

    lines = parse_xml.split('\n')
    with open(output_xml, 'w') as w:
        for line in lines:
            w.write(line)
            w.write('\n')
            print(line)

    return


def image_data_parse_to_xml(filename,
                              name,
                              width,
                              height,
                              depth,
                              xmin,
                              ymin,
                              xmax,
                              ymax,):
    """
    image data parse to xml file.
    input:
      filename: strings of filename
      names: strings of class category name
      height: height of image
      depth:  if image is RGB, depth is 3, if grayscale, depth is 1
      xmin: x1 coordinates of bounding box
      ymin: y1 coordinates of bounding box
      xmax: x2 coordinates of bounding box
      ymax: y2 coordinates of bounding box
    return:
      dom.toprettyxml()
    """
    dom = xml.dom.minidom.Document()
    annotation = dom.createElement('annotation')
    dom.appendChild(annotation)
    
    #------------#
    # root level #
    #------------#
    object_dom = dom.createElement('object')
    annotation.appendChild(object_dom)
    
    #--------------#
    # object level #
    #--------------#
    name_dom = dom.createElement('name')
    name_dom.appendChild(dom.createTextNode(str(name)))
    
    truncated_dom = dom.createElement('truncated')
    truncated_dom.appendChild(dom.createTextNode(str(0)))
    
    difficult_dom = dom.createElement('difficult')
    difficult_dom.appendChild(dom.createTextNode(str(0)))
    
    size_dom = dom.createElement('size')
    
    bndbox_dom = dom.createElement('bndbox')
    
    level_2 = [name_dom, truncated_dom, difficult_dom, size_dom, bndbox_dom]
    for i in level_2:
        object_dom.appendChild(i)
    
    #------------#
    # size level #
    #------------#
    width_dom = dom.createElement('width')
    width_dom.appendChild(dom.createTextNode(str(width)))
    
    height_dom = dom.createElement('height')
    height_dom.appendChild(dom.createTextNode(str(height)))
    
    depth_dom = dom.createElement('depth')
    depth_dom.appendChild(dom.createTextNode(str(depth)))
    
    level_size = [width_dom, height_dom, depth_dom]
    for i in level_size:
        size_dom.appendChild(i)
    
    #--------------#
    # bndbox level #
    #--------------#
    xmin_dom = dom.createElement('xmin')
    xmin_dom.appendChild(dom.createTextNode(str(xmin)))
    
    ymin_dom = dom.createElement('ymin')
    ymin_dom.appendChild(dom.createTextNode(str(ymin)))
    
    xmax_dom = dom.createElement('xmax')
    xmax_dom.appendChild(dom.createTextNode(str(xmax)))
    
    ymax_dom = dom.createElement('ymax')
    ymax_dom.appendChild(dom.createTextNode(str(ymax)))
    
    level_bndbox = [xmin_dom, ymin_dom, xmax_dom, ymax_dom]
    for i in level_bndbox:
        bndbox_dom.appendChild(i)
    
    # # debug print
    # print(dom.toprettyxml())

    return dom.toprettyxml()


def main(output_dir, original_dir):
    global img, break_flag
    
    # events = [i for i in dir(cv2) if 'EVENT' in i]
    # print(events)
    
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    image_dir = os.path.join(output_dir, 'image')
    if os.path.exists(image_dir) is False:
        os.mkdir(image_dir)
    annotation_dir = os.path.join(output_dir, 'annotation')
    if os.path.exists(annotation_dir) is False:
        os.mkdir(annotation_dir)
    
    images = []
    for root, dirs, files in os.walk(original_dir):
        targets = [os.path.join(root, f) for f in files]
        images.extend(targets)
    
    count = 1
    for image in images:
        # class name = directory name
        class_name = os.path.basename(os.path.dirname(image))
        # class name = filename header
        # class_name = os.path.basename(image).split('_')[0]
        print(image)
        print(class_name)
        
        output_image = os.path.join(image_dir, os.path.basename(image))
        shutil.copy2(image, output_image)
        
        img = cv2.imread(os.path.join(image_dir, image))
        print(img.shape)
        shape = img.shape
    
        filename = os.path.basename(image)
        names = class_name
        width = shape[0]
        height = shape[1]
        depth = shape[2]
        
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', callback)
    
        break_flag = 0
        print(str(count) +  '/'  + str(len(images)))
        while(True):
            cv2.imshow('image', img)
            k = cv2.waitKey(1)
            if k == 27:
                break
            if break_flag == 1:
                break
    
        cv2.destroyAllWindows()
    
        xmin = x1
        ymin = y1
        xmax = x2
        ymax = y2
        print('min', (x1, y1))
        print('max', (x2, y2))

        # image data parse to xml
        parse_xml = image_data_parse_to_xml(filename,
                                            names,
                                            width,
                                            height,
                                            depth,
                                            xmin,
                                            ymin,
                                            xmax,
                                            ymax,)
        # output xml
        generate_xml(filename, annotation_dir, parse_xml)
        
        count += 1

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        type=str,
                        default=None,
                        help='please enter xml output path',)
    parser.add_argument('--original_dir',
                        dest='original_dir',
                        type=str,
                        default=None,
                        help='please enter input image file path')
    argv = parser.parse_args()

    main(argv.output_dir, argv.original_dir)
    
