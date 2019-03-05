from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle
import threading
from datetime import datetime
from collections import deque

import requests
import numpy as np
import cv2
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

from nets import mobilenet_v1
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

_URL = 'http://localhost:8080/update_recipe'
# _LOG_DIR = '/media/panasonic/644E9C944E9C611A/tmp/log'

#---------------------#
# ingredient property #
#---------------------#
_INGREDIENT_NUM_CLASSES = 16
_INGREDIENT_DATA_DIR = 'slim/tfrecord/ingredient/'
_INGREDIENT_LABEL_DATA = 'labels.txt'
_INGREDIENT_DB_DATA = 'labels_db.txt'
_INGREDIENT_CHECKPOINT_PATH = 'slim/model/ingredient'
_INGREDIENT_CHECKPOINT_FILE = 'model.ckpt-20000'

#------------------#
# cooking property #
#------------------#
_COOKING_NUM_CLASSES = 6
_COOKING_DATA_DIR = 'object_detection_api/object_detection/data'
_COOKING_LABEL_DATA = 'gyoza_20190219_label_map.pbtxt'
# _COOKING_DB_DATA = 'labels_db.txt'
_COOKING_CHECKPOINT_PATH = 'object_detection_api/export/20190222_fix-lid_125833'
_COOKING_CHECKPOINT_FILE = 'frozen_inference_graph.pb'

#------------------#
# cookware property #
#------------------#
_COOKWARE_NUM_CLASSES = 4
_COOKWARE_DATA_DIR = 'slim/tfrecord/cookware'
_COOKWARE_LABEL_DATA = 'labels.txt'
# _COOKWARE_DB_DATA = 'labels_db.txt'
_COOKWARE_CHECKPOINT_PATH = 'slim/model/cookware'
_COOKWARE_CHECKPOINT_FILE = 'model.ckpt-20000'

# ------------- #
# cooking state #
# ------------- #
_COOKING_STATE = {
    0: 'フライパンに油をひき、中火で加熱します',
    1: '餃子を並べます',
    2: '水溶き片栗粉を回し入れます',
    3: '蓋をします',
    4: '中火で蒸し焼きにします',
    5: '水気がなくなってきたら、蓋を取ります',
    6: '火を強めて水分を飛ばします',
}

# ----------------------- #
# cooking state parameter #
# ----------------------- #
with open('object_detection_api/model/20190222_fix-lid_125833/1_1/1_1.pkl', 'rb') as m_1_1:
    model_1_1 = pickle.load(m_1_1)
with open('object_detection_api/model/20190222_fix-lid_125833/1_2/1_2.pkl', 'rb') as m_1_2:
    model_1_2 = pickle.load(m_1_2)
with open('object_detection_api/model/20190222_fix-lid_125833/2_1/2_1.pkl', 'rb') as m_2_1:
    model_2_1 = pickle.load(m_2_1)
with open('object_detection_api/model/20190222_fix-lid_125833/2_2/2_2.pkl', 'rb') as m_2_2:
    model_2_2 = pickle.load(m_2_2)
with open('object_detection_api/model/20190222_fix-lid_125833/3_1/3_1.pkl', 'rb') as m_3_1:
    model_3_1 = pickle.load(m_3_1)
with open('object_detection_api/model/20190222_fix-lid_125833/3_2/3_2.pkl', 'rb') as m_3_2:
    model_3_2 = pickle.load(m_3_2)
with open('object_detection_api/model/20190222_fix-lid_125833/4_1/4_1.pkl', 'rb') as m_4_1:
    model_4_1 = pickle.load(m_4_1)
with open('object_detection_api/model/20190222_fix-lid_125833/4_2/4_2.pkl', 'rb') as m_4_2:
    model_4_2 = pickle.load(m_4_2)
with open('object_detection_api/model/20190222_fix-lid_125833/5/5.pkl', 'rb') as m_5:
    model_5 = pickle.load(m_5)
with open('object_detection_api/model/20190222_fix-lid_125833/6/6.pkl', 'rb') as m_6:
    model_6 = pickle.load(m_6)

_COOKING_STATE_PARAM = {
    0: (model_1_1, model_1_2, model_2_1),
    1: (model_2_1, model_2_2, model_3_1),
    2: (model_3_1, model_3_2, model_4_1),
    3: (model_4_1, model_4_2, model_5),
    4: (model_5, model_5, model_5),
    5: (model_5, model_6, model_3_1),
    6: (model_3_1, model_3_1, model_1_1) # no state
}

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


class IngredientThread(threading.Thread):
  def __init__(self,
               bbox3,
               bbox4,
               # output_dir,
               checkpoint_file,
               category_map,
               db_map,
               previous_predictions_1,
               bbox3_instances,
               bbox4_instances,
               current_instances,
               bbox3_category,
               bbox4_category):
    super(IngredientThread, self).__init__()
    self.bbox3 = bbox3
    self.bbox4 = bbox4
    # self.output_dir = output_dir
    self.checkpoint_file = checkpoint_file
    self.category_map = category_map
    self.db_map = db_map
    self.previous_predictions_1 = previous_predictions_1
    self.current_predictions_1 = []
    self.bbox3_instance = bbox3_instances
    self.bbox4_instance = bbox4_instances
    self.current_instances = []
    self.bbox3_category = bbox3_category
    self.bbox4_category = bbox4_category
    
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
          num_classes=_INGREDIENT_NUM_CLASSES,
          is_training=False,
      )
      
    vars = slim.get_variables_to_restore()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      bbox3 = self.bbox3[:,:,::-1]
      bbox4 = self.bbox4[:,:,::-1]
      
      # evaluation
      log = str()
      all_bbox = [bbox3, bbox4]
      bbox_names = ['left', 'right']
      self.current_instances = []
      recognition_rates = []
      for bbox, bbox_name in zip(all_bbox, bbox_names):
        saver.restore(sess, self.checkpoint_file)
        x = end_points['Predictions'].eval(
            feed_dict={input: bbox}
        )
        # output top predicitons
        if bbox_name == 'left':
            print('*'*20 + 'LEFT' + '*'*20)
            self.current_instances.append(self.category_map[x.argmax()])
            category_name = str(self.category_map[x.argmax()])
            probability = '{:.4f}'.format(x.max())
            self.bbox3_instance = str(category_name) + ' : ' + probability
            recognition_rates.append(x.max())
            self.bbox3_category = self.category_map[x.argmax()]
        elif bbox_name == 'right':
            print('*'*20 + 'RIGHT' + '*'*20)
            self.current_instances.append(self.category_map[x.argmax()])
            category_name = str(self.category_map[x.argmax()])
            probability = '{:.4f}'.format(x.max())
            self.bbox4_instance = str(category_name) + ' : ' + probability
            recognition_rates.append(x.max())
            self.bbox4_category = self.category_map[x.argmax()]
        print(sys.stdout.write(
            '%s Top 1 prediction: %d %s %f'
            % (str(bbox_name), x.argmax(), self.category_map[x.argmax()], x.max())
        ))

        # output all class probabilities
        for i in range(x.shape[1]):
          print(sys.stdout.write('%s : %s' % (self.category_map[i], x[0][i])))

        pred_id = self.db_map[self.category_map[x.argmax()]]

        self.current_predictions_1.append(pred_id)

      try:
        if self.previous_predictions_1[0] is '':
          self.previous_predictions_1[0] = '0'
        if self.previous_predictions_1[1] is '':
          self.previous_predictions_1[1] = '0'
      except IndexError:
        print('IndexError')
        pass

      # send request when it detects changing ingredients
      ## if self.previous_predictions_1 != self.current_predictions_1:
      if self.previous_predictions_1 != self.current_predictions_1:
        print('change')
        print('*'*15 + ' request ' +'*'*15)
        t_query_0 = time.time()

        # send request when it detects gyoza
        # if self.category_map[x.argmax()] == 'gyoza':
        if self.bbox4_category == 'gyoza' or self.bbox3_category == 'gyoza':

          print('detect gyoza')
          print('take a picture')
          print('*'*30 + ' request ' +'*'*30)
          t_query_0 = time.time()
          query = 'http://localhost:8080/update_recipe?ingredient_ids1=ingredient_ids2=&frying_pan=gyoza'
          print('*'*15 + ' gyoza picture ' + '*'*15)
          print('gyoza query', query)
          requests.get(query)
          t_query_1 = time.time()
          print('request time :', t_query_1 - t_query_0)
        else:
          if self.current_predictions_1[0] == '0' and self.current_predictions_1[1] == '0':
            self.current_predictions_1[0] = '0'
            self.current_predictions_1[1] = '0'
          if self.current_predictions_1[0] == '0' and self.current_predictions_1[1] is not '0':
            self.current_predictions_1[0] = ''
          if self.current_predictions_1[1] == '0' and self.current_predictions_1[0] is not '0':
            self.current_predictions_1[1] = ''
          print('*'*15 + ' ingredient query ' + '*'*15)
          query = 'http://localhost:8080/update_recipe?ingredient_ids1={}&ingredient_ids2={}&frying_pan=false&page_index=0&ingredient_name1={}&ingredient_name2={}&recognition_rate1={:.4f}&recognition_rate2={:.4f}'.format(
              self.current_predictions_1[0],
              self.current_predictions_1[1],
              self.current_instances[0],
              self.current_instances[1],
              recognition_rates[0],
              recognition_rates[1],
          )
          print('*'*15 + 'query' + '*'*15, query)
          requests.get(query)
          t_query_1 = time.time()
          print('request time :', t_query_1 - t_query_0)
      else:
        print('not change')
        pass


class CookingThread(threading.Thread):
  def __init__(self,
               bbox2,
               # output_dir,
               checkpoint_file,
               category_map,
               # db_map,
               previous_state_num,
               previous_substate_num,
               previous_observation,
               cooking_state,
               cooking_state_param,
               bbox2_instance,):
    super(CookingThread, self).__init__()
    self.bbox2 = bbox2
    # self.output_dir = output_dir
    self.checkpoint_file = checkpoint_file
    self.category_map = category_map
    # self.db_map = db_map
    self.previous_state_num = previous_state_num
    self.previous_substate_num = previous_substate_num
    self.previous_observation = previous_observation
    self.cooking_state = cooking_state
    self.cooking_state_param = cooking_state_param
    self.current_state_num = previous_state_num
    self.current_substate_num = previous_substate_num
    self.previous_state = cooking_state[self.previous_state_num][self.previous_substate_num]
    self.previous_state_param = cooking_state_param[self.previous_state_num][self.previous_substate_num]
    self.num_classes = _COOKING_NUM_CLASSES
    self.bbox2_instance = str(self.previous_state_num) + ' : ' + str(cooking_state[self.previous_state_num])
 
  def kl_divergence(self, p, q, dx=0.001):
    p = p + dx
    q = q + dx

    return np.sum(p * (np.log(p / q)))

  def array_to_histogram(self, array, num_classes):
    histogram = np.array([array.count(c + 1) for c in range(num_classes)])

    return histogram / np.sum(histogram)
    
  def run(self):
    # ---------------- #
    # gyoza navigation #
    # ---------------- #
    print('gyoza navigation')
    observation = self.array_to_histogram(
        self.previous_observation, self.num_classes
    )
    print('observation')
    if self.previous_state == self.cooking_state[6]: # state finish
      pass
    else:
      print('previous state number')
      print(self.previous_substate_num)
      if self.previous_substate_num < 2:
        print('current_substate_num', self.previous_substate_num)
        current_distribution = self.cooking_state_param[self.previous_state_num][self.previous_substate_num]
        next_distribution = self.cooking_state_param[self.previous_state_num][self.previous_substate_num + 1]
        current_kl = self.kl_divergence(observation,  current_distribution)
        next_kl = self.kl_divergence(observation, next_distribution)
        print('================')
        print('current     : ', current_distribution)
        print('next        : ', next_distribution)
        print('observation : ', observation)
        print('current kl  : ', current_kl)
        print('next kl     : ', next_kl)
        if current_kl > next_kl:
          self.current_substate_num += 1
          print('################ substate num change {} -> {} ################'
                .format(self.previous_substate_num, self.current_substate_num))
        else:
          pass
      else:
        print('################ state change ################')
        self.current_state_num += 1
        self.current_substate_num = 0
        print('################ state change {} -> {} ################'
              .format(self.previous_state_num, self.current_state_num))
        query = 'http://localhost:8080/update_recipe?ingredient_ids1=&ingredient_ids2=&frying_pan={}'.format(
            self.current_state_num
        )
        requests.get(query)
        print('@@@@@@@@@@@@@@@@ cooking request throw @@@@@@@@@@@@@@@@')
        

class CookwareThread(threading.Thread):
  def __init__(self,
               bbox1,
               # output_dir,
               checkpoint_file,
               category_map,
               bbox1_instances,
               current_instances,
               ingredient_current_instances,
               bbox1_category):
    super(CookwareThread, self).__init__()
    self.bbox1 = bbox1
    # self.output_dir = output_dir
    self.checkpoint_file = checkpoint_file
    self.category_map = category_map
    self.bbox1_instance = bbox1_category
    self.current_instances = []
    self.ingredient_current_instances = ingredient_current_instances
    self.bbox1_category = bbox1_category
    self.kettle_flag = False

  def run(self):
    print('category_map', self.category_map)
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
          num_classes=_COOKWARE_NUM_CLASSES,
          is_training=False,
      )
      
    vars = slim.get_variables_to_restore()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      bbox1 = self.bbox1[:,:,::-1]
      
      # evaluation
      log = str()
      bbox = bbox1
      recognition_rates = []
      saver.restore(sess, self.checkpoint_file)
      x = end_points['Predictions'].eval(
          feed_dict={input: bbox}
      )
      # output top predicitons
      self.current_instances.append(self.category_map[x.argmax()])
      category_name = str(self.category_map[x.argmax()])
      probability = '{:.4f}'.format(x.max())
      self.bbox1_instance = str(category_name) + ' : ' + probability
      recognition_rates.append(x.max())
      if self.bbox1_category == self.category_map[x.argmax()]:
        self.kettle_flag = True
      self.bbox1_category = self.category_map[x.argmax()]
      print('self.bbox1_category', self.bbox1_category)
      print(sys.stdout.write(
          'Top 1 prediction: %d %s %f'
          % (x.argmax(), self.category_map[x.argmax()], x.max())
      ))
      
      # output all class probabilities
      for i in range(x.shape[1]):
        print(sys.stdout.write('%s : %s' % (self.category_map[i], x[0][i])))

      self.current_instances.append(x.argmax())

      # when ingredient name1 and 2 is nothing, throw request which instance is kettle or not
      print('self.kettle_flag', self.kettle_flag)
      if self.kettle_flag == False:
        print('self.ingredient_current_instances', self.ingredient_current_instances)
        if self.ingredient_current_instances[0] == 'nothing' \
          and self.ingredient_current_instances[1] == 'nothing':
          if self.category_map[x.argmax()] == 'kettle':
            print('detect kettle')
            print('*'*15 + ' request ' +'*'*15)
            t_query_0 = time.time()
            query = 'http://localhost:8080/update_kettle?kettle_exist=true'
            print('query', query)
            requests.get(query)
            t_query_1 = time.time()
            print('request time :', t_query_1 - t_query_0)
          else:
            print('not kettle')
            t_query_0 = time.time()
            query = 'http://localhost:8080/update_kettle?kettle_exist=false'
            print('query', query)
            requests.get(query)
            t_query_1 = time.time()
            print('request time :', t_query_1 - t_query_0)
            pass
        else:
          pass


def main():
  now = datetime.now()
  today = now.strftime('%Y%m%d')
  
  t0 = time.time()

  # output_dir = os.path.join(_LOG_DIR, today)
  # if os.path.isdir(output_dir) is False:
  #   os.mkdir(output_dir)
   
  #--------------#
  # define model #
  #--------------#
  ingredient_checkpoint_file = os.path.join(
      _INGREDIENT_CHECKPOINT_PATH, _INGREDIENT_CHECKPOINT_FILE
  )
  ingredient_category_map = convert_label_files_to_dict(
      _INGREDIENT_DATA_DIR, _INGREDIENT_LABEL_DATA
  )

  cooking_checkpoint_file = os.path.join(
      _COOKING_CHECKPOINT_PATH, _COOKING_CHECKPOINT_FILE
  )
  cooking_category_map = os.path.join(
      _COOKING_DATA_DIR, _COOKING_LABEL_DATA
  )

  cookware_checkpoint_file = os.path.join(
      _COOKWARE_CHECKPOINT_PATH, _COOKWARE_CHECKPOINT_FILE
  )
  cookware_category_map = convert_label_files_to_dict(
      _COOKWARE_DATA_DIR, _COOKWARE_LABEL_DATA
  )
  
  db_map = convert_label_files_to_dict(
      _INGREDIENT_DATA_DIR, _INGREDIENT_DB_DATA
  )

  # ------------------------- #
  # object detection property #
  # ------------------------- #
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(cooking_checkpoint_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      print(od_graph_def.ParseFromString(serialized_graph))
      tf.import_graph_def(od_graph_def, name='')
  cooking_label_map = label_map_util.load_labelmap(cooking_category_map)
  cooking_categories = label_map_util.convert_label_map_to_categories(
      cooking_label_map,
      max_num_classes=_COOKING_NUM_CLASSES,
  )
  cooking_category_index = label_map_util.create_category_index(cooking_categories)

  #-----------------------------#
  # videocapture and prediction #
  #-----------------------------#
  width = 1920
  height = 1080

  # define ROI
  threshold = int(275 / 2)                         # default (224 / 2)
  margin = 10                                      # not to capture bounding box

  center = int(width * 6.0/10)
  center1_width = int(center - (threshold*2 + margin)) # ROI1 center x
  center2_width = int(center)                          # ROI2 center x
  center3_width = int(center + (threshold*2 + margin)) # ROI2 center x
  center_height = int(height / 2)                  # ROI1,2 center y

  # print('center1_width :', center1_width)
  # print('center2_width :', center2_width)
  # print('center_height :', center_height)
  
  cap = cv2.VideoCapture(0)
  # cap = cv2.VideoCapture(
  #     '/media/panasonic/644E9C944E9C611A/tmp/data/mov/20180907/20180907_gyoza_kibun_hanetuski.mp4')

  # camera propety(1920x1080)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # start video capture
  count = 0

  frying_pan_flag = False
  is_state5 = False  
  previous_predictions_1 = []
  previous_observation = deque([1] * 40, maxlen=40)
  previous_state_num = 0
  previous_substate_num = 0
  cooking_state = _COOKING_STATE
  cooking_state_param = _COOKING_STATE_PARAM
  ingredient_current_instances = ['nothing', 'nothing']
  cookware_current_instances = []
  bbox1_category, bbox2_category, bbox3_category, bbox4_category = '','','',''
  bbox1_instance, bbox2_instance, bbox3_instance, bbox4_instance = 'bbox1', 'bbox2', 'bbox3', 'bbox4'

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      t1 = time.time()
      print('start ~ while :', t1 - t0)
      while(True):
        t3 = time.time()
        ret, frame = cap.read()
        share_margin = int(margin/2)
        text_margin = 10
        stings_space = 50

        # access category name for each thread class
        try:
          ingredient_current_instances = thread1.current_instances
        except UnboundLocalError:
          print('Unboundlocalerror')
          pass
        try:
          cookware_current_instances = thread3.current_instances
        except UnboundLocalError:
          print('Unboundlocalerror')
          pass          
          # print('thread1.current_instances', thread1.current_instances)
          # print('type', type(thread1.current_instances))
        try:
          bbox1_instance = thread3.bbox1_instance
        except UnboundLocalError:
          print('Unboundlocalerror')
          pass
        try:
          bbox2_instance = thread2.bbox2_instance
        except UnboundLocalError:
          print('Unboundlocalerror')
          pass
        try:
          bbox3_instance = thread1.bbox3_instance
        except UnboundLocalError:
          print('Unboundlocalerror')
          pass
        try:
          bbox4_instance = thread1.bbox4_instance
        except UnboundLocalError:
          print('Unboundlocalerror')
          pass
        try:
          bbox1_category = thread3.bbox1_category
        except UnboundLocalError:
          print('Unboundlocalerror')
          pass          

        # bbox2(cooking) --- for output with japanese ---
        message = bbox2_instance
        b,g,r,a = 0, 0, 255, 0
        fontpath = '/home/panasonic/.fonts/RictyDiminished-Bold.ttf'
        font = ImageFont.truetype(fontpath, 32)
        img_jp = Image.fromarray(frame)
        draw_jp = ImageDraw.Draw(img_jp)
        position = (center1_width-threshold-440,
                    (center_height+190)+(threshold)+(text_margin*3))
        draw_jp.text(position, message, font=font, fill=(b,g,r,a))
        frame = np.array(img_jp)        

        # # higobashi bbox
        # # bbox1(cookware)
        # cv2.rectangle(
        #     frame,
        #     ((center1_width-threshold-(share_margin)),
        #      (center_height-(threshold*3)-(share_margin))),
        #     ((center1_width+threshold+(share_margin)),
        #      (center_height-threshold-(share_margin))),
        #     (0,0,255),
        #     3
        # )
        # cv2.putText(
        #     frame,
        #     str(bbox1_instance),
        #     (center1_width-threshold, center_height-(threshold*3)-text_margin),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     2,
        #     (255,0,0),
        #     3,
        #     cv2.LINE_AA
        # )
        # # bbox2(cooking)
        # cv2.rectangle(
        #     frame,
        #     ((center1_width-threshold-(share_margin)),
        #      (center_height-threshold-(share_margin))), # start coordinates
        #     ((center1_width+threshold+(share_margin)),
        #      (center_height+threshold+(share_margin))), # end coordinates
        #     (0,0,255),
        #     3
        # )
        # cv2.putText(
        #     frame,
        #     str(count),
        #     (center1_width-threshold, center_height+threshold+text_margin+stings_space),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     2,
        #     (255,0,0),
        #     3,
        #     cv2.LINE_AA
        # )
        # # bbox3(ingredient1)
        # cv2.rectangle(
        #     frame,
        #     ((center2_width-threshold-(share_margin)),
        #      (center_height-threshold-(share_margin))), # start coordinates
        #     ((center2_width+threshold+(share_margin)),
        #      (center_height+threshold+(share_margin))), # end coordinates 
        #     (0,0,255),
        #     3
        # )
        # cv2.putText(
        #     frame,
        #     str(bbox3_instance),
        #     (center2_width-threshold, center_height-threshold-text_margin),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     2,
        #     (255,0,0),
        #     3,
        #     cv2.LINE_AA
        # )
        # # bbox4(ingredient2)
        # cv2.rectangle(
        #     frame,
        #     ((center3_width-threshold-(share_margin)),
        #      (center_height-threshold-(share_margin))), # start coordinates
        #     ((center3_width+threshold+(share_margin)),
        #      (center_height+threshold+(share_margin))), # end coordinates
        #     (0,0,255),
        #     3
        # )
        # cv2.putText(
        #     frame,
        #     str(bbox4_instance),
        #     (center3_width-threshold, center_height-threshold-text_margin),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     2,
        #     (255,0,0),
        #     3,
        #     cv2.LINE_AA
        # )
      
        # kusatsu bbox
        # bbox1(cookware)
        cv2.rectangle(
            frame,
            ((center1_width-threshold-(share_margin))-440,
             (center_height-threshold-(share_margin))+140), # start coordinates
            ((center1_width+threshold+(share_margin))-440,
             (center_height+threshold+(share_margin))+190), # end coordinates
            (0,0,255),
            3
        )
        cv2.putText(
            frame,
            str(bbox1_instance),
            (center1_width-threshold-440, (center_height+140)-(threshold)-(text_margin)-50),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            (0,0,255),
            3,
            cv2.LINE_AA
        )
        # # bbox2(cooking)
        # cv2.rectangle(
        #     frame,
        #     ((center1_width-threshold-(share_margin))-440,
        #      (center_height-threshold-(share_margin))+190), # start coordinates
        #     ((center1_width+threshold+(share_margin))-440,
        #      (center_height+threshold+(share_margin))+190), # end coordinates
        #     (0,0,255),
        #     3
        # )
        
        # bbox2 default kusatsu
        cv2.rectangle(
            frame,
            ((center1_width-threshold-(share_margin))-440,
             (center_height-threshold-(share_margin))+140), # start coordinates
            ((center1_width+threshold+(share_margin))-440,
             (center_height+threshold+(share_margin))+190), # end coordinates
            (0,0,255),
            3
        )
        # # bbox2 for 20180907 video
        # cv2.rectangle(
        #     frame,
        #     ((center1_width-threshold-(share_margin))-190,
        #      (center_height-threshold-(share_margin))+140),
        #     ((center1_width+threshold+(share_margin))-190,
        #      (center_height+threshold+(share_margin))+190),
        #     (0, 0, 255),
        #     3)

        # bbox3(ingredient1)
        cv2.rectangle(
            frame,
            ((center2_width-threshold-(share_margin))-300,
             (center_height-threshold-(share_margin))+130), # start coordinates
            ((center2_width+threshold+(share_margin))-300,
             (center_height+threshold+(share_margin))+130), # end coordinates 
            (0,0,255),
            3
        )
        cv2.putText(
            frame,
            str(bbox3_instance),
            ((center2_width-300)-threshold, (center_height+130)-threshold-text_margin),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            (0,0,255),
            3,
            cv2.LINE_AA
        )
        # bbox4(ingredient2)
        cv2.rectangle(
            frame,
            ((center3_width-threshold-(share_margin))-300,
             (center_height-threshold-(share_margin))+130), # start coordinates
            ((center3_width+threshold+(share_margin))-300,
             (center_height+threshold+(share_margin))+130), # end coordinates
            (0,0,255),
            3
        )
        cv2.putText(
            frame,
            str(bbox4_instance),
            ((center3_width-300)-threshold, (center_height+130)+threshold+text_margin*3),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            (0,0,255),
            3,
            cv2.LINE_AA
        )
      
        # cv2.setMouseCallback('frame', print_coordinates)
      
        # ROI
        # bbox1 = frame[center_height-threshold+190:center_height+threshold+190,
        #               center1_width-threshold-440:center1_width+threshold-440]
        # default
        bbox1 = frame[center_height-threshold+140:center_height+threshold+140,
                      center1_width-threshold-440:center1_width+threshold-440]
        # default kusatsu
        bbox2 = frame[center_height-threshold+190:center_height+threshold+190,
                      center1_width-threshold-440:center1_width+threshold-440]
        # bbox2 = frame[center_height-threshold+140:center_height+threshold+140,
        #               center1_width-threshold-440:center1_width+threshold-440]
        # # 20180907 video
        # bbox2 = frame[center_height-threshold+190:center_height+threshold+190,
        #               center1_width-threshold-190:center1_width+threshold-190]
        # default 
        bbox3 = frame[center_height-threshold+130:center_height+threshold+130,
                      center2_width-threshold-300:center2_width+threshold-300]
        # default
        bbox4 = frame[center_height-threshold+130:center_height+threshold+130,
                      center3_width-threshold-300:center3_width+threshold-300]
      
        # # save image of bounding box
        # now = datetime.now()
        # seconds = now.strftime('%Y%m%d_%H%M%S') + '_' + str(now.microsecond)
        # cv2.imwrite(os.path.join(output_dir, seconds) + '_bbox1.png', bbox1)
        # cv2.imwrite(os.path.join(output_dir, seconds) + '_bbox2.png', bbox2)
        # cv2.imwrite(os.path.join(output_dir, seconds) + '_bbox3.png', bbox3)
        # cv2.imwrite(os.path.join(output_dir, seconds) + '_bbox4.png', bbox4)
      
        # # BGR t0 RGB
        # bbox1 = bbox1[:,:,::-1]
        bbox2 = bbox2[:,:,::-1]
        # bbox3 = bbox3[:,:,::-1]
      
        # ---------------- #
        # object detection #
        # ---------------- #    
        _bbox = np.expand_dims(bbox2, 0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detection = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detection) = sess.run(
            [boxes, scores, classes, num_detection],
            feed_dict={image_tensor: _bbox},
        )
        vis_util.visualize_boxes_and_labels_on_image_array(
            bbox2,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            cooking_category_index,
            use_normalized_coordinates=True,
            line_thickness=20,
            max_boxes_to_draw=20,
        )
        # append top score category
        previous_observation.append(float(classes[0][0]))
        # # output cuurent objdect detection
        # print('current observation')
        # print(previous_observation[-1])
        # print('score')
        # print(scores[0][0])
        # TODO? log output

        # ---------- #
        # evaluation #
        # ---------- 
        # if count % 50 == 1:
        if count % 15 == 1:
          thread1 = IngredientThread(
              bbox3,
              bbox4,
              # output_dir,
              ingredient_checkpoint_file,
              ingredient_category_map,
              db_map,
              previous_predictions_1,
              bbox3_instance,
              bbox4_instance,
              ingredient_current_instances,
              bbox3_category,
              bbox4_category,
            )
          thread1.start()
          print('thread1', thread1)
          previous_predictions_1 = thread1.current_predictions_1

        if count % 15 == 6 and frying_pan_flag == True:
        # if count % 50 == 16:
          thread2 = CookingThread(
              bbox2,
              # output_dir,
              cooking_checkpoint_file,
              cooking_category_map,
              # db_map,
              previous_state_num,
              previous_substate_num,
              previous_observation,
              cooking_state,
              cooking_state_param,
              bbox2_instance,
            )
          thread2.start()
          thread2.join()                    
          print('thread2', thread2)
          # print('thread2.current_state_num', thread2.current_state_num)
          # print('thread2.current_substate_num', thread2.current_substate_num)
          previous_state_num = thread2.current_state_num
          previous_substate_num = thread2.current_substate_num
          if not is_state5 and thread2.current_state_num == 4: # mushiyaki
            print('~~~~~~~~~~~~~~~~ mushiyaki ~~~~~~~~~~~~~~~~')
            time_state5_start = time.time()
            is_state5 = True
          elif is_state5 and thread2.current_state_num == 4:
            print('~~~~~~~~~~~~~~~~ time ~~~~~~~~~~~~~~~~')              
            time_state5_now = time.time()
            print(time_state5_now - time_state5_start)            
            if time_state5_now - time_state5_start > 120:
              print('~~~~~~~~~~~~~~~~ state change ~~~~~~~~~~~~~~~~')
              previous_state_num += 1
              previous_substate_num = 0
            else:
              pass
       
        if count % 15 == 11:
        # if count % 50 == 33:
          thread3 = CookwareThread(
              bbox1,
              # output_dir,
              cookware_checkpoint_file,
              cookware_category_map,
              bbox1_instance,
              cookware_current_instances,
              ingredient_current_instances,
              bbox1_category,
            )
          thread3.start()
          # thread3.join()
          # print('thread3', thread3)
          bbox1_category = thread3.bbox1_category
          # print('bbox1 category')
          print(bbox1_category)
          if thread3.bbox1_category == 'frying-pan':
            frying_pan_flag = True

        t4 = time.time()
        # print('loop seconds :', t4 - t3)
                
        count += 1
        cv2.imshow('frame', frame)
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      
      cap.release()
      cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
