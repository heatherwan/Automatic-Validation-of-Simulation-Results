"""
DensePoint Layers in Tensorflow

Author: Wanting Lin
Date: May 2020

"""

import os

import tensorflow as tf

from utils import tf_util
from utils.cpp_modules import farthest_point_sample, gather_point, query_ball_point, group_point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def sample_and_group(npoint, radius, nsample, xyz, features, use_xyz=True):
    """
    Input:
        ===Parameters for query points=====
        npoint: int32 - points selected by farthest sampling
        radius: float32 - area for neighbor point search in local
        nsample: int32 - #points for neighbor point search in local
        ======= Varaibles ======================
        xyz: (batch_size, ndataset, 3) TF tensor
        features: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as features
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        grouped_features: (batch_size, npoint, nsample, 3+channel) TF tensor
    """

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    idx, _ = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization (xj-xi)

    if features is not None:
        grouped_features = group_point(features, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            grouped_features = tf.concat([grouped_xyz, grouped_features],
                                         axis=-1)  # (batch_size, npoint, nample, 3+channel)
    else:  # for first layer without any additional features
        grouped_features = grouped_xyz

    return new_xyz, grouped_features


def group_all(xyz, features, use_xyz=True):
    """
    concatenate xyz and other channels, and expand dimension for MLP
    Inputs:
        xyz: (batch_size, npoint, 3) TF tensor
        features: (batch_size, npoint, channel) TF tensor
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_features: (batch_size, 1, npoint, 3+channel) TF tensor
   """
    if use_xyz:
        new_features = tf.concat([xyz, features], axis=2)  # (batch_size, 16, 259)
    else:
        new_features = features
    new_features = tf.expand_dims(new_features, 1)  # (batch_size, 1, 16, 259)

    return new_features


def densepoint_module(xyz, features, is_training, bn_decay, scope=None, bn=True,
                      npoint=None, radius=None, nsample=None, mlp=None,
                      ppool=None, use_xyz=True, group_num=1):
    """ DensePoint module with PPool, Enhanced PConv and Global Pooling
        Input:
            xyz: (batch_size, npoint, 3) TF tensor
            features: (batch_size, npoint, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points selected in each local region
            (if selected point less than expected, duplicate the selected)
            mlp: int32 -- output size for SLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor -centroid points
            For PPool and Global Pooling:
                new_features: (batch_size, npoint, Cin+3) TF tensor
            For enhancedPConv:
                all_new_features : (batch_size, npoint, cin+3+Cout/4)
    """
    with tf.compat.v1.variable_scope(scope) as sc:

        if npoint is not None:
            new_xyz, grouped_features = sample_and_group(npoint, radius, nsample, xyz, features, use_xyz)
            if ppool:  # ppool
                grouped_features = tf_util.batch_norm_for_conv2d(grouped_features, is_training=is_training,
                                                                 scope='BNConcat', bn_decay=bn_decay)
                new_grouped_features = tf_util.conv2d(grouped_features, mlp, [1, 1],
                                                      padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                                      scope='PPool', bn_decay=bn_decay)
                new_features = tf.reduce_max(input_tensor=new_grouped_features, axis=[2])  # max pooling
                return new_xyz, new_features

            else:  # EnhancedPointConv
                # new_xyz: B N 3, grouped_features: B N S C+3
                # conv_phi function: grouped version
                new_grouped_features = tf_util.conv2d_group(grouped_features, mlp, kernel_size=[1, 1],
                                                            group_num=group_num,
                                                            padding='VALID', stride=[1, 1], bn=bn,
                                                            is_training=is_training,
                                                            scope='PhiConv', bn_decay=bn_decay)
                new_features = tf.reduce_max(input_tensor=new_grouped_features, axis=[2])  # max pooling

                # conv_psi
                new_features = tf_util.conv1d(new_features, mlp // 4, kernel_size=1,
                                              padding='VALID', stride=1, bn=bn, is_training=is_training,
                                              scope='PsiConv', bn_decay=bn_decay)

                # features: B N 1 Cin, B N 1 Cout/4 => B N 1 Cin+Cout/4
                all_new_features = tf.concat([features, new_features], axis=-1)
                # batch normalize the concatenate output
                all_new_features = tf_util.batch_norm_for_conv2d(all_new_features, is_training=is_training,
                                                                 scope='BNConcat', bn_decay=bn_decay)
                all_new_features = tf.nn.relu(all_new_features)

                return new_xyz, all_new_features
        # GlobalPooling
        else:
            grouped_features = group_all(xyz, features)
            grouped_features = tf_util.conv2d(grouped_features, mlp, [1, 1],
                                              padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                              scope='globalpooling', bn_decay=bn_decay)

            new_features = tf.reduce_max(input_tensor=grouped_features, axis=[2])  # max pooling
            return new_features
