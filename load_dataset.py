"""
Load datasets from TFRecords

__author__ = "MM. Kamani"
"""

import os
import numpy as np
import tensorflow as tf

class CifarDataset():

  def __init__(self,
              data_dir,
              subset='train',
              use_distortion=True,
              dataset='cifar10'):

    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion
    if dataset == 'cifar10':
      self.num_class = 10
    elif dataset == 'cifar100':
      self.num_class = 100
    self.WIDTH = 32
    self.HEIGHT = 32
    self.DEPTH = 3

  def get_filenames(self, subset):
    if subset in ['train', 'validation', 'eval']:
      return [os.path.join(self.data_dir, subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32) / 128.0 - 1
    image.set_shape([self.HEIGHT * self.WIDTH * self.DEPTH])
    
    image = tf.cast(tf.reshape(image, [self.HEIGHT, self.WIDTH, self.DEPTH]),tf.float32)
    image = self.preprocess(image)
    
    # label = tf.cast(tf.one_hot(features['label'], self.num_class), tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return image, label

  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    return self._create_tfiterator(batch_size, self.subset)


  def _create_tfiterator(self, batch_size, subset):
    filenames = self.get_filenames(subset=subset)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat()
    # Parse records.
    dataset= dataset.map(
      self.parser, num_parallel_calls=batch_size)

    # Ensure that the capacity is sufficiently large to provide good random
    # shuffling.
    dataset = dataset.shuffle(buffer_size = 3 * batch_size)

    # Batch it up.
    dataset= dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch 

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [self.HEIGHT, self.WIDTH, self.DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image
  
  @staticmethod
  def num_examples_per_epoch(subset='train', dataset='cifar10'):
    if dataset == 'cifar10':
      if subset == 'train':
        return 45000
      elif subset == 'validation':
        return 5000
      elif subset == 'eval':
        return 10000
      else:
        raise ValueError('Invalid data subset "%s"' % subset)
    elif dataset == 'cifar100':
      if subset == 'train':
        return 50000
      elif subset == 'validation':
        return 0
      elif subset == 'eval':
        return 10000
      else:
        raise ValueError('Invalid data subset "%s"' % subset)

