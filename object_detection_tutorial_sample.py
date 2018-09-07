import os
import sys
import tarfile
import six.moves.urllib as urllib
import tarfile
import zipfile
from io import StringIO
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from object_detection.utils import ops as util_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
# NUM_CLASSES = 90

MODEL_NAME = 'export'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(
    'object_detection/data', 'ingradient_label_map.pbtxt'
)
NUM_CLASSES = 3


# # download model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     print('file_name', file_name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())

# load a (frozen) tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fld:
        serialized_graph = fld.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
print('label_map', label_map)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True,
)
category_index = label_map_util.create_category_index(categories)
print('category_index', category_index)

# helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#-----------#
# detection #
#-----------#
# define test image file path 
PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'
TEST_IMAGE_PATHS = []
for root, dirs, files in os.walk(PATH_TO_TEST_IMAGES_DIR):
    for f in files:
        filepath = os.path.join(root, f)
        if filepath.find('.jpg') >= 0 or filepath.find('.png') >= 0:
            TEST_IMAGE_PATHS.append(filepath)
        else:
            print('{} is not jpg or png format file'.format(filepath))
            pass

IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs
            }
            # print('***************************************')
            # print('all_tensor_names %s' % all_tensor_names)
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes',
                        'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],[0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'],[0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0,0], [real_num_detection,-1])
                detection_masks = tf.slice(detection_masks, [0,0], [real_num_detection,-1,-1])
                detection_masks_reframed = util_ops.reframe_box_masks_to_image_masks(
                    detection_masks,
                    detection_boxes,
                    image.shape[0],
                    image.shape[1],
                )
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5),
                    tf.uint8,
                )
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed,
                    0,
                )

            image_tensor = tf.get_default_graph().get_tensor_by_name(
                'image_tensor:0'
            )
            output_dict = sess.run(
                tensor_dict,
                feed_dict={image_tensor: np.expand_dims(image, 0)}
            )

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict
                

for idx, image_path in enumerate(TEST_IMAGE_PATHS):
    print('idx', idx)
    print('test image', image_path)
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expand = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8
    )
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()
