"""ResNet model.
Create ResNet class from tf.keras.Model and BaseResNet

__author__ = "MM. Kamani"
"""
import tensorflow as tf
from model_base import BaseResNet



class ResNet(tf.keras.Model, BaseResNet):
  def __init__(self,
               num_layers,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_last',
               version='v1',
               num_class=10):
    super(ResNet,self).__init__()
    BaseResNet.__init__(self, is_training,data_format,batch_norm_decay,batch_norm_epsilon)
    
    self.n = (num_layers - 2) // 6
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_class = num_class
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
    
    self._model_layers = []
    #Build the model
    inputs = tf.placeholder(tf.float32, [None,32,32,3])
    x, conv1 = self._conv(inputs, 3, 16, 1)
    x, batch_norm1 = self._batch_norm(x)
    self._model_layers.append([conv1, batch_norm1])

    x = self._relu(x)

    # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
    if version == 'v1':
      self.res_func_build = self._residual_v1_build
      self.res_func = self._residual_v1
    elif version == 'v2':
      self.res_func_build = self._residual_v2_build
      self.res_func = self._residual_v2
    elif version == 'bv2':
      self.res_func_build = self._bottleneck_residual_v2_build
      self.res_func = self._bottleneck_residual_v2
  
    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x, res_layers = self.res_func_build(x, 3, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x, res_layers = self.res_func_build(x, 3, self.filters[i + 1], self.filters[i + 1], 1)
          self._model_layers.append(res_layers)

    x = self._global_avg_pool(x, True)
    x, dense1 = self._fully_connected(x, self.num_class)
    self._model_layers.append([dense1])

  def call(self, x, input_data_format):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])
    
    for l in self._model_layers[0]:
      x = l(x)
    x = self._relu(x)

    # 3 stages of block stacking.
    for i in range(3):
      for j in range(self.n):
        if j == 0:
          # First block in a stage, filters and strides may change.
          x = self.res_func(x, self.filters[i], self.filters[i + 1],
                        self._model_layers[1 + i*self.n])
        else:
          # Following blocks in a stage, constant filters and unit stride.
          x = self.res_func(x, self.filters[i + 1], self.filters[i + 1],
                        self._model_layers[1 + i*self.n + j])

    x = self._global_avg_pool(x)
    x = self._model_layers[-1][0](x)

    return x

