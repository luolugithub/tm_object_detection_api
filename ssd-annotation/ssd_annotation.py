import os
import shutil

import xml.dom.minidom

import numpy as np
import cv2

x1, y1, x2, y2 = 0, 0, 0, 0
filename = []
names = []
width, height, depth = [], [], []
xmin, ymin, xmax, ymax = [], [], [], []

def callback(event, x, y, flags, params):
    global x1, y1, x2, y2, xmin, ymin, xmax, ymax, img
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        xmin.append(x)
        ymin.append(y)
        x1 = x
        y1 = y
    elif event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        xmax.append(x)
        ymax.append(y)
        x2 = x
        y2 = y
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)


events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

data_dir = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/20180829_valid'
if os.path.exists(data_dir) is False:
    os.mkdir(data_dir)
image_dir = os.path.join(data_dir, 'image')
if os.path.exists(image_dir) is False:
    os.mkdir(image_dir)
annotation_dir = os.path.join(data_dir, 'annotation')
if os.path.exists(annotation_dir) is False:
    os.mkdir(annotation_dir)

original_data_dir = '/media/panasonic/644E9C944E9C611A/tmp/data/img/ssd_food_dossari_20180823_cu_ep_tm_valid'

images = []
for root, dirs, files in os.walk(original_data_dir):
    targets = [os.path.join(root, f) for f in files]
    images.extend(targets)

for image in images:
    class_name = os.path.basename(os.path.dirname(image))
    print(image)
    print(class_name)
    
    output_image = os.path.join(image_dir, os.path.basename(image))
    shutil.copy2(image, output_image)
    
    img = cv2.imread(os.path.join(image_dir, image))
    print(img.shape)
    shape = img.shape

    filename.append(os.path.basename(image))
    names.append(class_name)
    width.append(shape[0])
    height.append(shape[1])
    depth.append(shape[2])
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', callback)

    while(True):
        cv2.imshow('image', img)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()


print('filename', filename)
print('names', names)
print('width', width)
print('height', height)
print('depth', depth)
print('xmin', xmin)
print('ymin', ymin)
print('xmax', xmax)
print('ymax', ymax)


count = 0
truncated = 0
difficult = 0
for f,n,w,h,d,x1,y1,x2,y2 in zip(filename,
                                 names,
                                 width,
                                 height,
                                 depth,
                                 xmin,
                                 ymin,
                                 xmax,
                                 ymax,):
    
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
    name_dom.appendChild(dom.createTextNode(str(n)))
    
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
    width_dom.appendChild(dom.createTextNode(str(w)))
    
    height_dom = dom.createElement('height')
    height_dom.appendChild(dom.createTextNode(str(h)))
    
    depth_dom = dom.createElement('depth')
    depth_dom.appendChild(dom.createTextNode(str(d)))

    level_size = [width_dom, height_dom, depth_dom]
    for i in level_size:
        size_dom.appendChild(i)
    
    #--------------#
    # bndbox level #
    #--------------#
    xmin_dom = dom.createElement('xmin')
    xmin_dom.appendChild(dom.createTextNode(str(x1)))
    
    ymin_dom = dom.createElement('ymin')
    ymin_dom.appendChild(dom.createTextNode(str(y1)))
    
    xmax_dom = dom.createElement('xmax')
    xmax_dom.appendChild(dom.createTextNode(str(x2)))
    
    ymax_dom = dom.createElement('ymax')
    ymax_dom.appendChild(dom.createTextNode(str(y2)))
    
    level_bndbox = [xmin_dom, ymin_dom, xmax_dom, ymax_dom]
    for i in level_bndbox:
        bndbox_dom.appendChild(i)
    
    # # debug print
    # print(dom.toprettyxml())
    
    # TODO output xml
    xml_filename, _ = os.path.splitext(f)
    output_xml = os.path.join(annotation_dir, xml_filename + '.xml')

    lines = dom.toprettyxml().split('\n')
    with open(output_xml, 'w') as w:
        for line in lines:
            w.write(line)
            w.write('\n')
            print(line)

    count += 1
