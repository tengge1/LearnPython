# https://tensorflow.google.cn/tensorboard/tensorboard_profiling_keras

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from packaging import version

import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers

import numpy as np

print("TensorFlow version: ", tf.__version__)

device_name = tf.test.gpu_device_name()
if not tf.test.is_gpu_available():
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
