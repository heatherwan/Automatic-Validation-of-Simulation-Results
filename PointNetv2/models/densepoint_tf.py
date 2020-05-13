"""
This is a tensorflow implementation of densepoint
Reference code: https://github.com/Yochengliu/DensePoint
@author: Wanting Lin

"""
import tensorflow as tf
import numpy as np
import sys
import os
from utils import tf_util
from Parameters import Parameters
from utils.pointnet_tf_util import pointnet_sa_module, pointnet_sa_module_msg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
para = Parameters()


# Add input placeholder
def placeholder_inputs_other(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_other_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, para.dim - 3))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, pointclouds_other_pl, labels_pl


def get_model_other(point_cloud, pointclouds_other, is_training, bn_decay=None):
    """ Classification DensePoint, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0]  # .value
    end_points = {}
    # concatenate all features together
    # point_cloud = tf.concat(axis=2, values=[point_cloud, pointclouds_other])

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1, 0.2, 0.4], [16, 32, 128],
                                               [[32, 32, 64], [64, 64, 128], [64, 96, 128]], is_training, bn_decay,
                                               scope='layer1', use_nchw=True)
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.2, 0.4, 0.8], [32, 64, 128],
                                               [[64, 64, 128], [128, 128, 256], [128, 128, 256]], is_training, bn_decay,
                                               scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                              mlp=[256, 512, 1024], mlp2=None, group_all=True, is_training=is_training,
                                              bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    # TODO: change the input pf first layer

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