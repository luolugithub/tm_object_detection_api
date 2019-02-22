import os
import sys
import argparse

from collections import deque

import numpy as np

import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


_PATH_TO_CKPT = 'export/export_20190213_100762/frozen_inference_graph.pb'
_PATH_TO_LABELS = 'object_detection/data/gyoza_20190208_label_map.pbtxt'
_NUM_CLASSES = 7
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



state = {
    0: 'frying-pan wo oite abura wo hiki masu',
    1: 'gyoza wo narabe masu',
    2: 'katakuriko wo ire masu',
    3: 'futa wo shime masu',
    4: 'mushiyaki ni shimasu',
    5: 'futa wo tori masu',
    6: 'suibun wo tobasite kansei desu',
}

distribution = {
    0: (np.array([10,0,0,0,0,0,0]), np.array([0,0,8,0,0,0,2]), np.array([10,0,0,0,0,0,0])),
    1: (np.array([10,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,10]), np.array([10,0,0,0,0,0,0])),
    2: (np.array([10,0,0,0,0,0,0]), np.array([0,10,0,0,0,0,0]), np.array([10,0,0,0,0,0,0])),
    3: (np.array([10,0,0,0,0,0,0]), np.array([1,0,0,0,8,0,1]), np.array([0,0,0,0,10,0,0])),
    4: (np.array([0,0,0,0,10,0,0]), np.array([9,0,0,0,1,0,0]), np.array([10,0,0,0,0,0,0])),
    5: (np.array([10,0,0,0,0,0,0]), np.array([5,5,0,0,0,0,0]), np.array([10,0,0,0,0,0,0])),
    6: (np.array([0,0,10,0,0,0,0]), np.array([0,0,10,0,0,0,0]), np.array([0,0,10,0,0,0,0])),
}


def kl_divergence(p, q, dx=0.001):
    p = p + dx
    q = q + dx
    return np.sum(p * (np.log(p / q)))


def logging_histogram(df):
    print('1:', len(df[df == 1]))
    print('2:', len(df[df == 2]))
    print('3:', len(df[df == 3]))
    print('4:', len(df[df == 4]))
    print('5:', len(df[df == 5]))
    print('6:', len(df[df == 6]))
    print('7:', len(df[df == 7]))

    return


def array_to_histogram(array):
    histogram = np.array([
        array.count(1),
        array.count(2),
        array.count(3),
        array.count(4),
        array.count(5),
        array.count(6),
        array.count(7),
    ])

    return histogram




def main(start_offset, video_file):

    # -------------------------
    # video capture property
    # -------------------------    
    width = 1920
    height = 1080

    # # 20180907
    # threshold = int(400 / 2)  # default (224 / 2)
    # margin = 10  # not to capture bounding box

    # center_width = int(width / 2) - 300
    # center_height = int(height / 2) + 150


    # 20181012
    threshold = int(400 / 2)  # default (224 / 2)
    margin = 10  # not to capture bounding box

    center_width = int(width / 2) - 550
    center_height = int(height / 2) + 200

    # ----------------------------
    # gyoza navigation property
    # ----------------------------
    window_size = 10
    start_pos = 0
    end_pos = window_size
    kld_scores = []
    change_state_threshold = 3
    current_state_num = 0
    current_state = state[current_state_num]
    state_history = {}
    threshold_over_points = {}
    distribution_num = 0
    current_distribution = distribution[current_state_num][distribution_num]
    first_state = {start_pos: current_state}
    
    top_categories = deque([], maxlen=10)

    state_history.update(first_state)
    
    # ---------
    # Session
    # ---------    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            cap = cv2.VideoCapture(video_file)

            # camera propety(1920x1080)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)

            number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print('number_of_frames', number_of_frames)
            print('fps', fps)

            count = 0
            p, q = np.empty((0,10), int), np.empty((0,10), int)
            start_pos = fps * start_offset
            print('start_pos', start_pos)

            classes_log = np.empty((0,100), int)
            scores_log = np.empty((0,100), float)

            # -----------------
            # video capture
            # -----------------
            # while (True):
            for i in range(start_pos, number_of_frames):
                # camera property
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                frame = cv2.resize(frame, dsize=(1920, 1080))
                cv2.rectangle(frame, ((center_width - threshold - margin),
                                      (center_height - threshold - margin)),
                                     ((center_width + threshold + margin),
                                      (center_height + threshold + margin)),
                                     (0, 0, 255), 3)

                # ROI
                bbox = frame[center_height - threshold:center_height + threshold,
                             center_width - threshold:center_width + threshold]
                bbox = bbox[:, :, ::-1]
                _bbox = np.expand_dims(bbox, 0)

                # -------------------
                # object detection
                # -------------------                
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detection = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detection) = sess.run(
                    [boxes, scores, classes, num_detection],
                    feed_dict={image_tensor: _bbox},
                )
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #     bbox,
                #     np.squeeze(boxes),
                #     np.squeeze(classes).astype(np.int32),
                #     np.squeeze(scores),
                #     category_index,
                #     use_normalized_coordinates=True,
                #     line_thickness=20,
                #     max_boxes_to_draw=20,
                # )

                # print('a number of class : {} {}'.format(classes.shape[0], classes.shape[1]))
                # print('scores  : {}'.format(scores))
                # print('a number of score : {} {}'.format(scores.shape[0], scores.shape[1]))

                # print(sys.stdout.write('classes_log {}'.format(classes_log)))
                # print(sys.stdout.write('scores_log {}'.format(scores_log)))
                classes_log = np.append(classes_log, np.array([classes[0].astype(np.int32)]), axis=0)
                scores_log = np.append(scores_log, np.array([scores[0]]), axis=0)

                # --------------------
                # gyoza navigation
                # --------------------
                top_categories.append(int(classes[0][0]))
                
                if i % 10 == 0:
                    print('gyoza_navigation')
                    start_pos = i
                    end_pos = i + window_size
                    if len(p) == 0 and len(q) == 0:
                        p = np.ones(7)
                        q = np.ones(7)
                    else:
                        q = array_to_histogram(top_categories)
                        kl = kl_divergence(p, q)
                        print(p)
                        print(q)
                        print(kl)
                        kld_scores.append(kl)
                        p = q
                        
                        # if kl > change_state_threshold:
                        print(scores[0][0])
                        if kl != 0 and scores[0][0] > 0.3:
                            memo_distribution = {start_pos: (q, kl)}
                            threshold_over_points.update(memo_distribution)
                            if current_state == state[6]:
                                continue
                            else:
                                if distribution_num < 2:
                                    print('distribution_num', distribution_num)
                                    current_distribution = distribution[current_state_num][distribution_num]
                                    next_distribution = distribution[current_state_num][distribution_num + 1]
                                else:                                    
                                    print('################ state change ################')
                                    current_state = state[current_state_num + 1]
                                    current_state_num += 1
                                    distribution_num = 0
                                    memo_state = {start_pos: current_state}
                                    print(start_pos)
                                    print('################ state change {} {} ################'.format(
                                        current_state,
                                        distribution_num))
                                    continue
                                current_kl = kl_divergence(q, current_distribution)
                                next_kl = kl_divergence(q, next_distribution)
                                print('================')
                                print(start_pos)
                                print('current', current_distribution)
                                print('next', next_distribution)
                                print('q', q)
                                print('current kl', current_kl)
                                print('next kl', next_kl)
                                print('================')
                                if current_kl > next_kl:
                                    previous_distribution_num = distribution_num
                                    distribution_num += 1
                                    print('################ distribution num change {} -> {} ################'.format(
                                        previous_distribution_num,
                                        distribution_num))
                                else:
                                    pass


                cv2.putText(frame, '{} {}'.format(current_state_num, current_state), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                count += 1
                print(count)
                # print(current_state[current_state_num])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # output log
                    # np.savetxt('classes.csv', classes_log, delimiter=',')
                    # np.savetxt('scores.csv', scores_log, delimiter=',')
                    for k, v in threshold_over_points.items():
                        print(k, v)
                    for k, v in state_history.items():
                        print(k, v)
                    break

            cap.release()
            cv2.destroyAllWindows()
            # output log
            # np.savetxt('classes.csv', classes_log, delimiter=',')
            # np.savetxt('scores.csv', scores_log, delimiter=',')
            for k, v in threshold_over_points.items():
                print(k, v)
            for k, v in state_history.items():
                print(k, v)                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--start_offset',
        dest='start_offset',
        type=int,
        default=0,
        help='please enter start offset (second)'
        ' that you want to start movie')
    parser.add_argument(
        '--video_file',
        dest='video_file',
        type=str,
        default=None,
        help='please enter a video file(filepath)'
        ' that you want to test movie')
    argv = parser.parse_args()
    main(argv.start_offset, argv.video_file)
