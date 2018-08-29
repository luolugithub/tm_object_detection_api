# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl import flags

import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib

tf.app.flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
tf.app.flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
tf.app.flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
tf.app.flags.DEFINE_integer('num_eval_steps', None, 'Number of train steps.')
tf.app.flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
tf.app.flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
    'one round of eval vs running continuously (default).'
)
tf.app.flags.DEFINE_boolean('eval_training_data', True,
                     'If training data should be evaluated for this job.')
FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
  # flags.mark_flag_as_required('model_dir')
  # flags.mark_flag_ads_required('pipeline_config_path')
  if not FLAGS.model_dir:
    raise ValueError('You must supply the model_dir')
  if not FLAGS.pipeline_config_path:
    raise ValueError('You must supply the pipeline_config_path')

  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
  print('config', config)

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      eval_steps=FLAGS.num_eval_steps)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fn = train_and_eval_dict['eval_input_fn']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']
  eval_steps = train_and_eval_dict['eval_steps']
  # print('estimator', estimator)
  # print('train_input_fn', train_input_fn)
  # print('eval_input_fn', eval_input_fn)
  # print('eval_on_train_input_fn', eval_on_train_input_fn)
  # print('predict_input_fn', predict_input_fn)
  # print('train_steps', train_steps)
  # print('eval_steps', eval_steps)


  if FLAGS.checkpoint_dir:
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      input_fn = eval_input_fn
    if FLAGS.run_once:
      estimator.evaluate(input_fn,
                         eval_steps,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.model_dir, input_fn,
                                eval_steps, train_steps, name)
  else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fn,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_steps,
        eval_on_train_data=False)
    print('train_spec', train_spec)
    print('eval_spec', eval_specs)


    print('estimator', estimator)
    print('train_spec', train_spec)
    print('eval_specs[0]', eval_specs[0])

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
  tf.app.run()
