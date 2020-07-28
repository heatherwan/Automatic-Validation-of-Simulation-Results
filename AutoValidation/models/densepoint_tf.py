"""
This is a tensorflow implementation of densepoint
Reference code: https://github.com/Yochengliu/DensePoint
@author: Wanting Lin

"""
import os
import sys

import tensorflow as tf

from utils.Parameters import Parameters
from utils import tf_util
from utils.densepoint_tf_util import densepoint_module

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

    # concatenate all features together SF,x,y,z, distance minSF
    point_cloud_SF = tf.expand_dims(point_cloud[:, :, 0], axis=-1)
    l0_xyz = point_cloud[:, :, 1:4]
    l0_points = tf.concat(axis=2, values=[point_cloud_SF, point_cloud[:, :, 4:para.dim]])

    # first stage: 1 PPool, 3 EnhancedPConv
    all_xyz, all_points = densepoint_module(l0_xyz, l0_points, is_training, bn_decay,
                                            npoint=512, radius=0.25, nsample=64, mlp=para.k_add * 4,
                                            scope='PPool1', ppool=True)

    # second stage: 2 PPool, 3 EnhancedPConv
    all_xyz, all_points = densepoint_module(all_xyz, all_points, is_training, bn_decay,
                                            npoint=256, radius=0.27, nsample=64, mlp=para.k_add * 4 - 3,
                                            scope='PPool2', ppool=True)

    for i in range(4):  # B 128 1 93 -> 24
        all_xyz, all_points = densepoint_module(all_xyz, all_points, is_training, bn_decay,
                                                npoint=256, radius=0.32, nsample=16, mlp=para.k_add * 4,
                                                group_num=para.group_num,
                                                scope=f'PConv1_{i + 1}')

    # # second stage: 2 PPool, 3 EnhancedPConv
    # all_xyz, all_points = densepoint_module(all_xyz, all_points, is_training, bn_decay,
    #                                         npoint=128, radius=0.32, nsample=64, mlp=para.k_add * 4 - 3,
    #                                         scope='PPool3', ppool=True)
    #
    # for i in range(4):  # B 128 1 93 -> 24
    #     all_xyz, all_points = densepoint_module(all_xyz, all_points, is_training, bn_decay,
    #                                             npoint=128, radius=0.39, nsample=16, mlp=para.k_add * 4,
    #                                             group_num=para.group_num,
    #                                             scope=f'PConv2_{i + 1}')

    l3_points = densepoint_module(all_xyz, all_points, is_training, bn_decay,
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


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Change the label from an integer to the one_hot vector.
    labels = tf.one_hot(indices=label, depth=para.outputClassN)

    # # without smoothing
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)

    # with smoothing
    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred,
                                                     label_smoothing=para.loss_smoothing)

    mean_classify_loss = tf.reduce_mean(input_tensor=loss)
    tf.compat.v1.summary.scalar('classify loss', mean_classify_loss)

    # make coarse label
    coarse_label = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(labels),
                                                             tf.constant([0, 1, 1, 1]), num_segments=2))
    pred_prob = tf.nn.softmax(pred)
    coarse_prob = tf.matmul(pred_prob, para.fine_coarse_mapping)
    coarse_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=coarse_label, logits=coarse_prob)
    mean_coarse_loss = tf.reduce_mean(input_tensor=coarse_loss)
    tf.compat.v1.summary.scalar('coarse loss', mean_coarse_loss)

    tf.compat.v1.summary.scalar('all loss', mean_classify_loss + mean_coarse_loss)

    if para.binary_loss:
        return mean_classify_loss + mean_coarse_loss
    else:
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
    # print(f'Total parameters number is {total_parameters}')
    return total_parameters
