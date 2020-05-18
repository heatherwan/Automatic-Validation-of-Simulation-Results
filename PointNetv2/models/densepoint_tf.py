"""
This is a tensorflow implementation of densepoint
Reference code: https://github.com/Yochengliu/DensePoint
@author: Wanting Lin

"""
import os
import sys

import tensorflow as tf

from Parameters import Parameters
from utils import tf_util
from utils.densepoint_tf_util import pointnet_sa_module_msg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
para = Parameters()


# Add input placeholder
def placeholder_inputs_other(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, para.dim))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, labels_pl


def get_model_other(point_cloud, is_training, bn_decay=None):
    """
        DesnsePoint with 2 PPools + 3 PConvs + 1 global pooling narrowness k = 24; group number g = 2

    """
    batch_size = point_cloud.get_shape()[0]  # .value
    end_points = {}
    # concatenate all features together

    l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])  # start point, len
    l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, para.dim - 3])  # start from fourth dimension

    # first stage: 1 PPool, 0 EnhancedPConv
    # In: B 1024 1 3, B 1024 1 6-3
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, is_training, bn_decay,
                                               npoint=512, radius=0.25, nsample=64, mlp=96,
                                               scope='PPool1', ppool=True)
    # Out: B 512 1 3, B 512 1 96
    # second stage: 1 PPool, 3 EnhancedPConv
    # B 128 1 3, B 128 1 93
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, is_training, bn_decay,
                                               npoint=128, radius=0.32, nsample=64, mlp=93,
                                               scope='PPool2', ppool=True)

    all_xyz = l2_xyz
    all_points = l2_points

    k = 24
    for i in range(3):  # B 128 1 93 -> 24
        all_xyz, all_points = pointnet_sa_module_msg(all_xyz, all_points, is_training, bn_decay,
                                                     npoint=128, radius=0.39, nsample=16, mlp=96,
                                                     scope=f'PConv{i + 1}', pooling_no=i)

    l3_points = pointnet_sa_module_msg(all_xyz, all_points, is_training, bn_decay,
                                       mlp=512, scope='GloPool')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, para.outputClassN, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss_weight(pred, label, end_points, classweight):
    """ pred: B*NUM_CLASSES,
      label: B, """
    labels = tf.one_hot(indices=label, depth=para.outputClassN)
    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    loss = tf.multiply(loss, classweight)  # multiply class weight with loss for each object

    mean_classify_loss = tf.reduce_mean(input_tensor=loss)
    tf.compat.v1.summary.scalar('classify loss', mean_classify_loss)

    return mean_classify_loss


def get_para_num():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim
        total_parameters += variable_parametes
    print(f'Total parameters number is {total_parameters}')
