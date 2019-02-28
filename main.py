"""main
main file to create the training process using tf estimator API.

__author__ = "MM. Kamani"
"""


from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import json
from collections import namedtuple

import load_dataset as ld
import model
import utils
import numpy as np
import six
from six.moves import xrange 
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def get_model_fn(features, labels, mode, params):
	"""Returns a function that will build the ResNet and apply it to the input"""

	"""Model body.

	Args:
		features: a list of tensors
		labels: a list of tensors
		mode: ModeKeys.TRAIN or EVAL
		params: Hyperparameters suitable for tuning
	Returns:
		A EstimatorSpec object.
	"""
	is_training = (mode == tf.estimator.ModeKeys.TRAIN)

	# channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
	# on CPU. The exception is Intel MKL on CPU which is optimal with
	# channels_last.
	num_gpus = len(utils.get_available_gpus())
	data_format = params.data_format
	if not data_format:
		if num_gpus == 0:
			data_format = 'channels_last'
		else:
			data_format = 'channels_first'
	if params.dataset == 'cifar10':
		num_class=10
	elif params.dataset == 'cifar100':
		num_class=100

	train_op = []
	
	# Building the main model
	with tf.variable_scope('resnet') as var_scope:
		model_loss, model_gradvars, model_preds = _model_fn(is_training,
																												params.weight_decay, 
																												features, 
																												labels, 
																												data_format,
              																					params.num_layers,
																												num_class, 
																												params.batch_norm_decay, 
																												params.batch_norm_epsilon, 
																												var_scope.name, 
																												params.version)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, var_scope.name)

  # Updating parameters
	# Suggested learning rate scheduling from
	# https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar100-resnet.py#L155
	num_batches_per_epoch = ld.CifarDataset.num_examples_per_epoch(
			'train', params.dataset) // params.train_batch_size
	boundaries = [
			num_batches_per_epoch * x
			for x in np.array([82, 123, 300], dtype=np.int64)
	]
	staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]

	learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
																						boundaries, staged_lr, name='learning_rate')

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op.append(
			optimizer.apply_gradients(
				model_gradvars, global_step=tf.train.get_global_step())
		)

	
	examples_sec_hook = utils.ExamplesPerSecondHook(
		params.train_batch_size, every_n_steps=100)

	tensors_to_log = {'Main loss':model_loss, 'learning_rate':learning_rate}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=100)
	train_hooks = [logging_hook, examples_sec_hook]


	train_op.extend(update_ops)
	train_op = tf.group(*train_op)


	accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), model_preds['classes'])

	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=model_preds,
		loss=model_loss,
		train_op=train_op,
		training_hooks=train_hooks,
		eval_metric_ops=metrics)


def _model_fn(is_training, weight_decay, feature, label, data_format,
              num_layers, num_class, batch_norm_decay, batch_norm_epsilon, scope, version):
	"""Build computation tower (Resnet).

	Args:
		is_training: true if is training graph.
		weight_decay: weight regularization strength, a float.
		feature: a Tensor.
		label: a Tensor.
		data_format: channels_last (NHWC) or channels_first (NCHW).
		num_layers: number of layers, an int.
		num_class: int, based on the number of output classes in each dataset.
		batch_norm_decay: decay for batch normalization, a float.
		batch_norm_epsilon: epsilon for batch normalization, a float.
		scope: is the scope name that this tower is building its graph on
		version: The version of the ResNet model.

	Returns:
		A tuple with the loss for the tower, the gradients and parameters, and
		predictions.

	"""
	resnet_model = model.ResNet(num_layers,
															is_training,
															batch_norm_decay,
															batch_norm_epsilon,
															data_format,
															version,
															num_class)
	resnet_logits = resnet_model(feature, data_format)
	model_preds = {
	  'classes': tf.argmax(input=resnet_logits, axis=1),
	  'probabilities': tf.nn.softmax(resnet_logits)
  }
	model_params = tf.trainable_variables(scope=scope)
	model_loss = tf.losses.softmax_cross_entropy(
		  logits=resnet_logits, onehot_labels=label)  
	model_loss +=  weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model_params])
	
	param_grads = tf.gradients(model_loss, model_params)
	return model_loss, zip(param_grads, model_params), model_preds


def input_fn(data_dir,
			 subset,
			 batch_size,
			 dataset='cifar10',
			 use_distortion_for_training=False):
	"""Create input graph for model.

	Args:
	data_dir: Directory where TFRecords representing the dataset are located.
	subset: one of 'train', 'validate' and 'eval'.
	batch_size: total batch size for training
	dataset: choices between 'cifar10', 'cifar100'
	use_distortion_for_training: True to use distortions.
	Returns:
	two tensors for features and labels
	"""
	with tf.device('/cpu:0'):
		use_distortion = (subset == 'train') and use_distortion_for_training
		d = ld.CifarDataset(data_dir=data_dir, subset=subset, use_distortion=use_distortion, dataset=dataset)
		feature, label = d.make_batch(batch_size)
		
	return feature, label



def main(job_dir, data_dir, use_distortion_for_training,
			 log_device_placement, **hparams):

  # Session configuration.
  sess_config = tf.ConfigProto(
	  allow_soft_placement=True,
	  log_device_placement=log_device_placement,
	  gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = utils.RunConfig(
	  session_config=sess_config, model_dir=job_dir)
  # config = config.replace(save_checkpoints_steps=100)

  train_input_fn = functools.partial(
	  input_fn,
	  data_dir,
	  subset='train',
	  batch_size=hparams['train_batch_size'],
		dataset=hparams['dataset'],
	  use_distortion_for_training=use_distortion_for_training)

  eval_input_fn = functools.partial(
	  input_fn,
	  data_dir,
	  subset='eval',
		dataset=hparams['dataset'],
	  batch_size=hparams['eval_batch_size'])

  

  train_steps = hparams['train_steps']
  eval_steps = ld.CifarDataset.num_examples_per_epoch('eval') // hparams['eval_batch_size']

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps,  throttle_secs=600)

  classifier = tf.estimator.Estimator(
	  model_fn=get_model_fn,
	  config=config,
	  params=tf.contrib.training.HParams(
            is_chief=config.is_chief,
                **hparams))

  # Create experiment.
  tf.estimator.train_and_evaluate(
	  estimator=classifier,
	  train_spec=train_spec,
	  eval_spec=eval_spec)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-dir',
		type=str,
		required=True,
		help='The directory where the CIFAR-10 input data is stored.')
	parser.add_argument(
		'--job-dir',
		type=str,
		required=True,
		help='The directory where the model will be stored.')
	parser.add_argument(
		'--num-layers',
		type=int,
		default=20,
		help='The number of layers of the model.')
	parser.add_argument(
		'--train-steps',
		type=int,
		default=80000,
		help='The number of steps to use for training.')
	parser.add_argument(
		'--train-batch-size',
		type=int,
		default=128,
		help='Batch size for training.')
	parser.add_argument(
		'--eval-batch-size',
		type=int,
		default=100,
		help='Batch size for validation.')
	parser.add_argument(
		'--momentum',
		type=float,
		default=0.9,
		help='Momentum for MomentumOptimizer.')
	parser.add_argument(
		'--weight-decay',
		type=float,
		default=2e-3,
		help='Weight decay for convolutions.')
	parser.add_argument(
		'--learning-rate',
		type=float,
		default=0.001,
		help="""\
		This is the inital learning rate value. The learning rate will decrease
		during training. For more details check the model_fn implementation in
		this file.\
		""")
	parser.add_argument(
		'--use-distortion-for-training',
		type=bool,
		default=True,
		help='If doing image distortion for training.')
	parser.add_argument(
		'--data-format',
		type=str,
		default=None,
		help="""\
		If not set, the data format best for the training device is used. 
		Allowed values: channels_first (NCHW) channels_last (NHWC).\
		""")
	parser.add_argument(
		'--log-device-placement',
		action='store_true',
		default=False,
		help='Whether to log device placement.')
	parser.add_argument(
		'--batch-norm-decay',
		type=float,
		default=0.997,
		help='Decay for batch norm.')
	parser.add_argument(
		'--batch-norm-epsilon',
		type=float,
		default=1e-5,
		help='Epsilon for batch norm.')
	parser.add_argument(
		'--dataset',
		type=str,
		choices=['cifar10','cifar100'],
		default='cifar10',
		help='Datset name to run the experiment on.'
	)
	parser.add_argument(
		'--version',
		type=str,
		choices=['v1','v2','bv2'],
		default='v1',
		help='Version of the ResNet model.'
	)

	args = parser.parse_args()

	if (args.num_layers - 2) % 6 != 0:
		raise ValueError('Invalid --num-layers parameter.')


	main(**vars(args))
