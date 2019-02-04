import os
import sys

import numpy as np

import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

_PATH_TO_CKPT = 'export/export_20190205_21247/frozen_inference_graph.pb'
_PATH_TO_LABELS = 'object_detection/data/gyoza_20190203_00-04_label_map.pbtxt'
_NUM_CLASSES = 5
# _PATH_TO_CKPT = './ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
# _PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
# _NUM_CLASSES = 90

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

filepath = '/media/tsukudamayo/0CE698DCE698C6FC/tmp/data/mov/cooking/gyoza/myself/20180926_003448.mp4'


def main():

    width = 1920
    height = 1080

    threshold = int(350 / 2)  # default (224 / 2)
    margin = 10  # not to capture bounding box

    center_width = int(width / 2) - 50
    center_height = int(height / 2) + 275

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            cap = cv2.VideoCapture(filepath)

            # camera propety(1920x1080)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)

            count = 0
            while (True):
                ret, frame = cap.read()
                cv2.rectangle(frame, ((center_width - threshold - margin),
                                      (center_height - threshold - margin)),
                              ((center_width + threshold + margin),
                               (center_height + threshold + margin)),
                              (0, 0, 255), 3)

                # ROI
                bbox = frame[center_height - threshold:center_height +
                             threshold, center_width - threshold:center_width +
                             threshold]
                # _bbox = cv2.resize(bbox,(224,224))
                bbox = bbox[:, :, ::-1]
                # _bbox = bbox / 128 - 1 # normalization
                _bbox = np.expand_dims(bbox, 0)

                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detection = detection_graph.get_tensor_by_name(
                    'num_detections:0')

                if count % 5 == 0:
                    (boxes, scores, classes, num_detection) = sess.run(
                        [boxes, scores, classes, num_detection],
                        feed_dict={image_tensor: _bbox},
                    )
                    print(
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            bbox,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=20,
                            max_boxes_to_draw=20,
                        ))
                    print(sys.stdout.write('classes : %s' % classes))

                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
