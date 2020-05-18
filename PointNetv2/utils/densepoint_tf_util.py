""" DensePoint Layers

Author: Wanting Lin
Date: May 2020

"""

import os
import sys

from utils.cpp_modules import farthest_point_sample, gather_point, query_ball_point, group_point, knn_point, three_nn, \
    three_interpolate

import tensorflow as tf
import numpy as np
from utils import tf_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def sample_and_group(npoint, radius, nsample, xyz, features, use_xyz=True):
    """
    Input:
        ===Parameters for query points=====
        npoint: int32
        radius: float32
        nsample: int32
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
        xyz: (batch_size, ndataset, 3) TF tensor
        features: (batch_size, ndataset, channel) TF tensor
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_features: (batch_size, 1, ndataset, 3+channel) TF tensor
   """
    if use_xyz:
        new_features = tf.concat([xyz, features], axis=2)  # (batch_size, 16, 259)
    else:
        new_features = features
    new_features = tf.expand_dims(new_features, 1)  # (batch_size, 1, 16, 259)

    return new_features


def pointnet_sa_module_msg(xyz, features, is_training, bn_decay, scope=None, bn=True,
                           npoint=None, radius=None, nsample=None, mlp=None,
                           ppool=None, pooling_no=None, use_xyz=True):
    """ PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            features: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_features: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    """
    with tf.compat.v1.variable_scope(scope) as sc:

        if npoint is not None:
            new_xyz, grouped_features = sample_and_group(npoint, radius, nsample, xyz, features, use_xyz)
            if ppool:
                new_grouped_features = tf_util.conv2d(grouped_features, mlp, [1, 1],
                                                      padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                                      scope='PPool', bn_decay=bn_decay)
                new_features = tf.reduce_max(input_tensor=new_grouped_features, axis=[2])  # max pooling
                return new_xyz, new_features

            else:  # EnhancedPointConv
                # new_xyz: B N 3, grouped_features: B N S C+3
                # conv_phi function: grouped version
                # TODO: implement group convolution
                new_grouped_features = tf_util.conv2d_group(grouped_features, mlp, kernel_size=[1, 1], group_num=4,
                                                            padding='VALID', stride=[1, 1], bn=bn,
                                                            is_training=is_training,
                                                            scope='PhiConv', bn_decay=bn_decay)
                print(new_grouped_features.get_shape())
                # conv_psi
                new_grouped_features = tf_util.conv1d(new_grouped_features, mlp // 4, kernel_size=1,
                                                      padding='VALID', stride=1, bn=bn, is_training=is_training,
                                                      scope='PsiConv', bn_decay=bn_decay)
                new_features = tf.reduce_max(input_tensor=new_grouped_features, axis=[2])  # max pooling
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


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    """ PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    """
    with tf.compat.v1.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum(input_tensor=(1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1
