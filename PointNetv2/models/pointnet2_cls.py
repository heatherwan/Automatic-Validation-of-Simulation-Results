"""
This is a modified for more features implementation of pointnet2
Reference code: https://github.com/charlesq34/pointnet2/
@author: Wanting Lin

"""
import os
import sys

import tensorflow as tf

from Parameters import Parameters
from utils import tf_util
from utils.pointnet_tf_util import pointnet_sa_module, pointnet_sa_module_msg

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
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
    """

    batch_size = point_cloud.get_shape()[0]  # .value
    end_points = {}

    l0_xyz = point_cloud[:, :, 1:4]
    l0_points = tf.concat(axis=2, values=[point_cloud[:, :, 0], point_cloud[:, :, 4:para.dim]])

    # Set abstraction layers
    # input B 1024 1 3 => 64+128+128 = 320  max pooling in small group n = 16 32 128
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1, 0.2, 0.4], [16, 32, 128],
                                               [[32, 32, 64], [64, 64, 128], [64, 96, 128]], is_training, bn_decay,
                                               scope='layer1')  # , use_nchw=True
    # input B 512 320 => 128+256+256 = 640  max pooling in small group n = 32 64 128
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.2, 0.4, 0.8], [32, 64, 128],
                                               [[64, 64, 128], [128, 128, 256], [128, 128, 256]], is_training, bn_decay,
                                               scope='layer2')

    # input B 128 640 => 1024, max pooling in all pointcloud = 128
    # MLP layer to gather 3 scale features
    _, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                         mlp=[256, 512, 1024], mlp2=None, group_all=True, is_training=is_training,
                                         bn_decay=bn_decay, scope='layer3')

    # input B 1 1024
    # Fully connected layers

    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
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
