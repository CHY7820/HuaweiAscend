import sys
sys.path.insert(0, '')
sys.path.append('..')

from graph.kinetics import Graph
import tensorflow as tf
import numpy as np

INITIALIZER = None
REGULARIZER = None
# INITIALIZER = tf.variance_scaling_initializer(scale=2.,mode="fan_out",distribution="truncated_normal")
# REGULARIZER = tf.contrib.layers.l2_regularizer(0.0001)


class stgcn():
  def __init__(self, num_class):
    self.num_class = num_class  # 识别种类
    self.graph = Graph()
    # 将定义好的图矩阵A转化成tf.Variable，放在计算图中，但设置成不可训练
    self.A0 = tf.Variable(
        self.graph.
        A[0],  # 像在Graph()文件中说的那样，.A 和 .get_adjacency_matrix() 都可以获得A
        dtype=tf.float32,
        trainable=False,
        name='adjacency_matrix')
    self.A1 = tf.Variable(
        self.graph.
        A[1],  # 像在Graph()文件中说的那样，.A 和 .get_adjacency_matrix() 都可以获得A
        dtype=tf.float32,
        trainable=False,
        name='adjacency_matrix')
    self.A2 = tf.Variable(
        self.graph.
        A[2],  # 像在Graph()文件中说的那样，.A 和 .get_adjacency_matrix() 都可以获得A
        dtype=tf.float32,
        trainable=False,
        name='adjacency_matrix')

  def unit_gcn(self, x, fliter, is_training, subgraph, activation=tf.nn.sigmoid, downsample=False, residual=True):
    with tf.name_scope('gcn'):
      convx_sum = None
      for i in range(subgraph):
        convx = tf.layers.conv2d(inputs=x,
                                 filters=fliter,
                                 kernel_size=[1, 1],
                                 padding='same',
                                 activation=None,
                                 kernel_initializer=INITIALIZER,
                                 kernel_regularizer=REGULARIZER,
                                 data_format='channels_last')
        N = convx.get_shape()[0]
        T = convx.get_shape()[1]
        V = convx.get_shape()[2]
        C = convx.get_shape()[3]

        convx = tf.reshape(convx, [-1, V])
        if i == 0:
          convx = tf.matmul(convx, self.A0)
        if i == 1:
          convx = tf.matmul(convx, self.A1)
        if i == 2:
          convx = tf.matmul(convx, self.A2)
        convx = tf.reshape(convx, [-1, T, V, C])
        convx_sum = convx + convx_sum if convx_sum is not None else convx
      print('***************convx_sum1************', convx_sum)
      convx_sum = tf.layers.batch_normalization(convx_sum, training=is_training, axis=-1)  # axis=+1: NCHW
      print('***************convx_sum2************', convx_sum)
      return activation(convx_sum)

  def unit_tcn(self, x, fliter, is_training, kernel_size_t, stride, activation=tf.nn.sigmoid):
    with tf.name_scope('tcn'):
      convx = tf.layers.conv2d(inputs=x,
                               filters=fliter,
                               kernel_size=[kernel_size_t, 1],
                               strides=[stride, 1],
                               padding='same',
                               activation=None,
                               kernel_initializer=INITIALIZER,
                               kernel_regularizer=REGULARIZER,
                               data_format='channels_last')
      convx = tf.layers.batch_normalization(convx, training=is_training, axis=-1)
    return convx  # tcn后本来就不需要进行激活，因为之后还要有residual

  def unit_gcn_tcn(self, x, fliter, is_training, subgraph=3, kernel_size_t=9, stride=1, residual=True, downsample=False, activation=tf.nn.sigmoid, name='None'):
    with tf.name_scope(name=name):
      if not residual:
        residualx = 0
      elif residual and stride == 1 and not downsample:
        residualx = x
      else:
        residualx = tf.layers.conv2d(inputs=x,
                                     filters=fliter,
                                     kernel_size=[1, 1],
                                     strides=[stride, 1],
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=INITIALIZER,
                                     kernel_regularizer=REGULARIZER,
                                     data_format='channels_last')
        residualx = tf.layers.batch_normalization(residualx, training=is_training, axis=-1)
      x = self.unit_gcn(x,
                        fliter=fliter,
                        is_training=is_training,
                        subgraph=subgraph,
                        activation=activation,
                        residual=residual,
                        downsample=downsample)
      x = self.unit_tcn(x,
                        fliter=fliter,
                        is_training=is_training,
                        kernel_size_t=kernel_size_t,
                        stride=stride,
                        activation=activation)
      x = x + residualx
      return activation(x)

  def call(self, x, is_training=False):
    N = x.get_shape()[0]
    T = x.get_shape()[1]
    V = x.get_shape()[2]
    C = x.get_shape()[3]

    activation = tf.nn.relu  # 卷积网络的激活函数为relu
    with tf.name_scope('l1'):
      l1 = self.unit_gcn_tcn(x,
                             64,
                             is_training,
                             residual=False,
                             activation=activation,
                             name='l1')
    with tf.name_scope('l2'):
      l2 = self.unit_gcn_tcn(l1,
                             64,
                             is_training,
                             activation=activation,
                             name='l2')
    with tf.name_scope('l3'):
      l3 = self.unit_gcn_tcn(l2,
                             64,
                             is_training,
                             activation=activation,
                             name='l3')
    with tf.name_scope('l4'):
      l4 = self.unit_gcn_tcn(l3,
                             64,
                             is_training,
                             activation=activation,
                             name='l4')
    with tf.name_scope('l5'):
      l5 = self.unit_gcn_tcn(l4,
                             128,
                             is_training,
                             stride=2,
                             downsample=True,
                             activation=activation,
                             name='l5')
    with tf.name_scope('l6'):
      l6 = self.unit_gcn_tcn(l5,
                             128,
                             is_training,
                             activation=activation,
                             name='l6')
    with tf.name_scope('l7'):
      l7 = self.unit_gcn_tcn(l6,
                             128,
                             is_training,
                             activation=activation,
                             name='l7')
    with tf.name_scope('l8'):
      l8 = self.unit_gcn_tcn(l7,
                             256,
                             is_training,
                             stride=2,
                             downsample=True,
                             activation=activation,
                             name='l8')
    with tf.name_scope('l9'):
      l9 = self.unit_gcn_tcn(l8,
                             256,
                             is_training,
                             activation=activation,
                             name='l9')
    with tf.name_scope('l10'):
      l10 = self.unit_gcn_tcn(l9,
                              256,
                              is_training,
                              activation=activation,
                              name='l10')

    layers_out = l10

    layers_out_floot = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(layers_out)
    print('***************layers_out_floot************', layers_out_floot)
    # drop_out = tf.layers.dropout(layers_out_floot, rate=0, training=is_training)
    out = tf.layers.dense(layers_out_floot, self.num_class, name='output_logits')
    print('***************out************', out)

    return out
