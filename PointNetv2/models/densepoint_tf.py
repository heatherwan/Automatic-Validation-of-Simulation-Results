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
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, para.dim))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, labels_pl


# DensePoint: 2 PPools + 3 PConvs + 1 global pool; narrowness k = 24; group number g = 2
def get_model_other(point_cloud, is_training, bn_decay=None):
    """
            PointNet2 with multi-scale grouping
            Semantic segmentation network that uses feature propogation layers

            Parameters
            ----------
            num_classes: int
                Number of semantics classes to predict over -- size of softmax classifier that run for each point
            input_channels: int = 6
                Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
                value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
            use_xyz: bool = True
                Whether or not to use the xyz position of a point as a feature
        """
    """ Classification DensePoint, input is BxNx3, output Bx4 """
    batch_size = point_cloud.get_shape()[0]  # .value
    end_points = {}
    # concatenate all features together

    # l0_xyz = point_cloud
    # l0_points = None
    l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
    l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 3])
    print('ls shape ', l0_xyz.get_shape(), ' ', l0_points.get_shape())

    # Set abstraction layers
    # input B 1024 1 3 => 64+128+128 = 320  max pooling in small group n = 16 32 128
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1, 0.2, 0.4], [16, 32, 128],
                                               [[32, 32, 64], [64, 64, 128], [64, 96, 128]], is_training, bn_decay,
                                               scope='layer1')  # , use_nchw=True
    print('ls shape ', l1_xyz.get_shape(), ' ', l1_points.get_shape())
    # input B 512 320 => 128+256+256 = 640  max pooling in small group n = 32 64 128
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.2, 0.4, 0.8], [32, 64, 128],
                                               [[64, 64, 128], [128, 128, 256], [128, 128, 256]], is_training, bn_decay,
                                               scope='layer2')

    print('ls shape ', l2_xyz.get_shape(), ' ', l2_points.get_shape())
    # input B 128 640 => 1024, max pooling in all pointcloud = 128
    # MLP layer to gather 3 scale features
    _, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                         mlp=[256, 512, 1024], mlp2=None, group_all=True, is_training=is_training,
                                         bn_decay=bn_decay, scope='layer3')
    print('ls shape ', l3_points.get_shape())

    # input B 1 1024
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
