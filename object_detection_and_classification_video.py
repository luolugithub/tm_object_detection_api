from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime
import threading
import requests

import numpy as np
import cv2
import tensorflow as tf

from nets import mobilenet_v1

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


#-------------------------#
# classification property #
#-------------------------#
_URL_CLASSIFICATION = 'http://localhost:8080/update_recipe'
_NUM_CLASSES_CLASSIFICATION = 3
_DATA_DIR_CLASSIFICATION = '/media/panasonic/644E9C944E9C611A/tmp/data/tfrecord/food_224_dossari_20180815_cu_ep_tm_x10'
_LABEL_DATA_CLASSIFICATION = 'labels.txt'
_DB_DATA_CLASSIFICATION = 'labels_db.txt'
_CHECKPOINT_PATH_CLASSIFICATION = '/media/panasonic/644E9C944E9C611A/tmp/model/20180815_food_dossari_cu_ep_tm_x10_mobilenet_v1_1_224_finetune'
_CHECKPOINT_FILE_CLASSIFICATION = 'model.ckpt-20000'
_LOG_DIR = '/media/panasonic/644E9C944E9C611A/tmp/log'

#---------------------------#
# object detection property #
#---------------------------#
_CHECKPOINT_PATH_DETECTION = 'export/export_20180913'
_CHECKPOINT_FILE_DETECTION = 'frozen_inference_graph.pb'
_LABEL_DATA_DETECTION = 'object_detection/data/cooking_20180912_label_map.pbtxt'
_NUM_CLASSES_DETECTION = 3



def send_get_request(url, key, value):
    req = urllib.request.Request(
        '{}?{}'.format(
            url,
            urllib.parse.urlencode({key: value}))
    )
    urllib.request.urlopen(req)

def convert_label_files_to_dict(data_dir, label_file):
    category_map = {}
    keys, values = [], []
    
    # read label file
    with open(os.path.join(data_dir, label_file)) as f:
        lines = f.readlines()
        f.close()

    # label file convert into python dictionary
    for line in lines:
        key_value = line.split(':')
        try:
            key = int(key_value[0])
        except KeyError:
            key = key_value[0]
        except ValueError:
            key = str(key_value[0])
        value = key_value[1].split()[0] # delete linefeed
        category_map[key] = value
    
    return category_map


def print_coordinates(event, x, y, flags, param):
  """get the coordinates when left mouse button clicked"""
  print(x, y)


def settings_property():
    # #--------------------#
    # # property of opencv #
    # #--------------------#
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    # contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    # saturation = cap.get(cv2.CAP_PROP_SATURATION)
    # hue = cap.get(cv2.CAP_PROP_HUE)
    # # gain = cap.get(cv2.CAP_PROP_GAIN) # gain is not supported 
    # exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    # rectification = cap.get(cv2.CAP_PROP_RECTIFICATION)

    # monochrome = cap.get(cv2.CAP_PROP_MONOCHROME)
    # sharpness = cap.get(cv2.CAP_PROP_SHARPNESS)
    # auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    # gamma = cap.get(cv2.CAP_PROP_GAMMA)
    # temperture = cap.get(cv2.CAP_PROP_TEMPERATURE)
    # white_blance = cap.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)
    # zoom = cap.get(cv2.CAP_PROP_ZOOM)
    # focus = cap.get(cv2.CAP_PROP_FOCUS)
    # guid = cap.get(cv2.CAP_PROP_GUID)
    # iso_speed = cap.get(cv2.CAP_PROP_ISO_SPEED)
    # backlight = cap.get(cv2.CAP_PROP_BACKLIGHT)
    # pan = cap.get(cv2.CAP_PROP_PAN)
    # tilt = cap.get(cv2.CAP_PROP_TILT)
    # roll = cap.get(cv2.CAP_PROP_IRIS)
    
    # #----------------#
    # # property debug #
    # #----------------#
    # print('width', width)
    # print('height', height)
    # print('fps', fps)
    # print('brightness', brightness)
    # print('contrast', contrast)
    # print('saturation', saturation)
    # print('hue', hue)
    # # print('gain', gain) # gain is not supported
    # print('exposure', exposure)
    # print('hue', hue)
    # print('exposure', exposure)
    # print('rectification', rectification)
    # print('monochrome', monochrome)
    # print('sharpness', sharpness)
    # print('auto_exposure', auto_exposure)
    # print('gamma', gamma)
    # print('temperture', temperture)
    # print('white_blance', white_blance)
    # print('zoom', zoom)
    # print('focus', focus)
    # print('guid', guid)
    # print('iso_speed', iso_speed)
    # print('backlight', backlight)
    # # print('pan', pan)
    # # print('tilt', tilt)
    # # print('roll', roll)
    return


class ClassficationThread(threading.Thread):
  def __init__(self,
               bbox2,
               bbox3,
               checkpoint_file,
               category_map,
               db_map,
               previous_predictions,):
    super(ClassficationThread, self).__init__()
    self.bbox2 = bbox2
    self.bbox3 = bbox3
    self.checkpoint_file = checkpoint_file
    self.category_map = category_map
    self.db_map = db_map
    self.previous_predictions = previous_predictions
    self.current_predictions = []
    
  def run(self):
    tf.reset_default_graph()
    
    # file_input = tf.placeholder(tf.string, ())
    # input = tf.image.decode_png(tf.read_file(file_input))
    input = tf.placeholder('float', [None,None,3])
    images = tf.expand_dims(input, 0)
    images = tf.cast(images, tf.float32)/128 - 1
    images.set_shape((None,None,None,3))
    images = tf.image.resize_images(images,(224,224))

    with tf.contrib.slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
      logits, end_points = mobilenet_v1.mobilenet_v1(
          images,
          num_classes=_NUM_CLASSES_CLASSIFICATION,
          is_training=False,
      )
      
    vars = slim.get_variables_to_restore()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      bbox2 = self.bbox2 / 128 - 1
      bbox3 = self.bbox3 / 128 - 1
      bbox2 = np.expand_dims(bbox2, 0)
      bbox3 = np.expand_dims(bbox3, 0)
      
      # evaluation
      log = str()
      all_bbox = [bbox2, bbox3]
      bbox_names = ['left', 'right']
      for bbox, bbox_name in zip(all_bbox, bbox_names):
        saver.restore(sess, self.checkpoint_file)
        x = end_points['Predictions'].eval(
            feed_dict={images: bbox}
        )
        # output top predicitons
        if bbox_name == 'left':
            print('*'*20 + 'LEFT' + '*'*20)
        elif bbox_name == 'right':
            print('*'*20 + 'RIGHT' + '*'*20)
        print(sys.stdout.write(
            '%s Top 1 prediction: %d %s %f'
            % (str(bbox_name), x.argmax(), self.category_map[x.argmax()], x.max())
        ))

        # output all class probabilities
        for i in range(x.shape[1]):
          print(sys.stdout.write('%s : %s' % (self.category_map[i], x[0][i])))

        pred_id = self.db_map[self.category_map[x.argmax()]]

        self.current_predictions.append(pred_id)

        # send GET request if prediction is changed
      print('self.previous_predictions', self.previous_predictions)
      print('self.current_predictions', self.current_predictions)
      print(self.previous_predictions == self.current_predictions)
      if self.previous_predictions != self.current_predictions:
        print('change')
        t_query_0 = time.time()
        query = 'http://localhost:8080/update_recipe?ingredient_ids1={}&ingredient_ids2={}&flying_pan=true'.format(
            self.current_predictions[0],
            self.current_predictions[1],
        )
        # # query = 'http://localhost:3000?ingredient_ids={0}&ingredient_ids2={1}&flying_pan=true'.format(
        # #     current_predictions[0],
        # #     current_predictions[1],
        # # )
        requests.get(query)
        t_query_1 = time.time()
        print('request time :', t_query_1 - t_query_0)
      else:
        print('not change')
        pass


class ObjectDetectionThread(threading.Thread):

  def __init__(self,
               bbox1,
               detection_checkpoint_file,
               detection_category_map,
               detection_category_index):
    super(ObjectDetectionThread, self).__init__()
    self.bbox1 = bbox1
    self.checkpoint_file = detection_checkpoint_file
    self.category_map = detection_category_map
    self.category_index = detection_category_index

  def run(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self.checkpoint_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        print(od_graph_def.ParseFromString(serialized_graph))
        tf.import_graph_def(od_graph_def, name='')

        with tf.Session(graph=detection_graph) as sess:
          bbox = self.bbox1
          _bbox = np.expand_dims(bbox, 0)

          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detection = detection_graph.get_tensor_by_name('num_detections:0')

          (boxes, scores, classes, num_detection) = sess.run(
              [boxes, scores, classes, num_detection],
              feed_dict = {image_tensor: _bbox},
          )
          print(vis_util.visualize_boxes_and_labels_on_image_array(
              self.bbox1,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              self.category_index,
              use_normalized_coordinates=True,
              line_thickness=20,
              max_boxes_to_draw=20,
          ))

#           # print((boxes, scores, classes, num_detection))

#           # cv2.imshow('frame', self.frame)

def main():
  now = datetime.now()
  today = now.strftime('%Y%m%d')
  
  t0 = time.time()

  output_dir = os.path.join(_LOG_DIR, today)
  if os.path.isdir(output_dir) is False:
    os.mkdir(output_dir)
   
  #--------------#
  # define model #
  #--------------#
  # for ClassficationThread
  classfication_checkpoint_file = os.path.join(
      _CHECKPOINT_PATH_CLASSIFICATION, _CHECKPOINT_FILE_CLASSIFICATION
  )
  classification_category_map = convert_label_files_to_dict(
      _DATA_DIR_CLASSIFICATION, _LABEL_DATA_CLASSIFICATION
  )
  classification_db_map = convert_label_files_to_dict(
      _DATA_DIR_CLASSIFICATION, _DB_DATA_CLASSIFICATION
  )

  # for object_detection_thread
  detection_checkpoint_file = os.path.join(
      _CHECKPOINT_PATH_DETECTION, _CHECKPOINT_FILE_DETECTION
  )
  detection_label_map = label_map_util.load_labelmap(_LABEL_DATA_DETECTION)
  detection_categories = label_map_util.convert_label_map_to_categories(
      detection_label_map,
      max_num_classes=_NUM_CLASSES_DETECTION,
  )
  detection_category_index = label_map_util.create_category_index(
      detection_categories
  )
  

  #-----------------------------#
  # videocapture and prediction #
  #-----------------------------#
  width = 1920
  height = 1080

  # define ROI
  threshold = int(224 / 2)                         # default (224 / 2)
  margin = 10                                      # not to capture bounding box

  center = int(width * 6.0/10)
  center1_width = int(center - (threshold*2 + margin)) # ROI1 center x
  center2_width = int(center)                          # ROI2 center x
  center3_width = int(center + (threshold*2 + margin)) # ROI3 center x
  center_height = int(height / 2)                      # ROI1,2,3 center y

  print('center1_width :', center1_width)
  print('center2_width :', center2_width)
  print('center3_width :', center3_width)
  print('center_height :', center_height)
  
  cap = cv2.VideoCapture(0)

  # camera propety(1920x1080)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # requests.get('http://localhost:8080/update_recipe?ingredient_ids1=42,46&ingredient_ids2=43,617&frying_pan=true&page_index=0')
  
  # start video capture
  count = 0
  previous_predictions = []
  t1 = time.time()
  print('start ~ while :', t1 - t0)
  while(True):
    t3 = time.time()
    ret, frame = cap.read()

    share_margin = int(margin/2) # settings not to eval rectangle color
    cv2.rectangle(
        frame,
        ((center1_width-threshold-(share_margin)),
         (center_height-threshold-(share_margin))), # start coordinates
        ((center1_width+threshold+(share_margin)),
         (center_height+threshold+(share_margin))), # end coordinates
        (0,0,255),
        3
    )
    cv2.rectangle(
        frame,
        ((center2_width-threshold-(share_margin)),
         (center_height-threshold-(share_margin))), # start coordinates
        ((center2_width+threshold+(share_margin)),
         (center_height+threshold+(share_margin))), # end coordinates
        (0,0,255),
        3
    )
    cv2.rectangle(
        frame,
        ((center3_width-threshold-(share_margin)),
         (center_height-threshold-(share_margin))), # start coordinates
        ((center3_width+threshold+(share_margin)),
         (center_height+threshold+(share_margin))), # end coordinates
        (0,0,255),
        3
    )

    # cv2.setMouseCallback('frame', print_coordinates)

    # ROI
    bbox1 = frame[center_height-threshold:center_height+threshold,
                  center1_width-threshold:center1_width+threshold]
    bbox2 = frame[center_height-threshold:center_height+threshold,
                  center2_width-threshold:center2_width+threshold]
    bbox3 = frame[center_height-threshold:center_height+threshold,
                  center3_width-threshold:center3_width+threshold]
    
    # bbox1 = cv2.resize(bbox1,(224,224))
    # bbox2 = cv2.resize(bbox2,(224,224))
    # bbox3 = cv2.resize(bbox3,(224,224))
    
    # save image of bounding box
    now = datetime.now()
    seconds = now.strftime('%Y%m%d_%H%M%S') + '_' + str(now.microsecond)
    cv2.imwrite(os.path.join(output_dir, seconds) + '_bbox1.png', bbox1)
    cv2.imwrite(os.path.join(output_dir, seconds) + '_bbox2.png', bbox2)
    cv2.imwrite(os.path.join(output_dir, seconds) + '_bbox3.png', bbox3)

    if count % 21 == 0:
      thread_1 = ClassficationThread(
          bbox2,
          bbox3,
          classfication_checkpoint_file,
          classification_category_map,
          classification_db_map,
          previous_predictions,
        )
      thread_1.start()
      previous_predictions = thread_1.current_predictions
    elif count % 23 == 0:
      thread_2 = ObjectDetectionThread(
          bbox1,
          detection_checkpoint_file,
          detection_categories,
          detection_category_index,
      )
      thread_2.start()
    else:
      pass
      # detection_graph = tf.Graph()
      # with detection_graph.as_default():
      #   od_graph_def = tf.GraphDef()
      #   with tf.gfile.GFile(detection_checkpoint_file, 'rb') as fid:
      #     serialized_graph = fid.read()
      #     od_graph_def.ParseFromString(serialized_graph)
      #     print(od_graph_def.ParseFromString(serialized_graph))
      #     tf.import_graph_def(od_graph_def, name='')
      
      #     with tf.Session(graph=detection_graph) as sess:
      #       _bbox = np.expand_dims(bbox1, 0)
      
      #       image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      #       boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      #       scores = detection_graph.get_tensor_by_name('detection_scores:0')
      #       classes = detection_graph.get_tensor_by_name('detection_classes:0')
      #       num_detection = detection_graph.get_tensor_by_name('num_detections:0')
      
      #       (boxes, scores, classes, num_detection) = sess.run(
      #           [boxes, scores, classes, num_detection],
      #           feed_dict = {image_tensor: _bbox},
      #       )
      #       print(vis_util.visualize_boxes_and_labels_on_image_array(
      #           bbox1,
      #           np.squeeze(boxes),
      #           np.squeeze(classes).astype(np.int32),
      #           np.squeeze(scores),
      #           detection_category_index,
      #           use_normalized_coordinates=True,
      #           line_thickness=20,
      #           max_boxes_to_draw=20,
      #       ))

          # print((boxes, scores, classes, num_detection))
      
          
      
    t4 = time.time()
    print('loop seconds :', t4 - t3)
            
    count += 1

    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     bbox1,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     self.category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=20,
    #     max_boxes_to_draw=20,
    # )

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
