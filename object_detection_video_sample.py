import os, sys

import numpy as np
import cv2
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


_PATH_TO_CKPT = './ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
_PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
_NUM_CLASSES = 90


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(_PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        print(od_graph_def.ParseFromString(serialized_graph))
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(_PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=_NUM_CLASSES,
)
category_index = label_map_util.create_category_index(categories)

def main():

  width = 1920
  height = 1080

  threshold = int(300 / 2) # default (224 / 2)
  margin = 10              # not to capture bounding box

  center_width = int(width / 2)
  center_height = int(height / 2)

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

      cap = cv2.VideoCapture(0)
      
      # camera propety(1920x1080)
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
      
      while(True):
        ret, frame = cap.read()
        cv2.rectangle(
            frame,
            ((center_width-threshold-margin),(center_height-threshold-margin)),
            ((center_width+threshold+margin),(center_height+threshold+margin)),
            (0,0,255),
            3
        )

        frame = cv2.resize(frame, (width, height))
      
        # ROI
        # bbox = frame[center_height-threshold:center_height+threshold,
        #              center_width-threshold:center_width+threshold]
      
        frame_ex = frame / 128 - 1 # normalization
        # frame = cv2.resize(frame,(224,224))
        frame_ex = np.expand_dims(frame_ex, 0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detection = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detection) = sess.run(
            [boxes, scores, classes, num_detection],
            feed_dict = {image_tensor: frame_ex},
        )
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=10
        )
        
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

      cap.release()
      cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
