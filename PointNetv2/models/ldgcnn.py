"""
The network model of linked dynamic graph CNN. We borrow the edge 
convolutional operation from the DGCNN and design our own network architecture.
Reference code: https://github.com/WangYueFt/dgcnn
@author: Kuangen Zhang

"""
import tensorflow as tf
import numpy as np
import sys
import os
from utils import tf_util
from Parameters import Parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
para = Parameters()

# Add input placeholder
def placeholder_inputs_other(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_other_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, para.dim - 3))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, pointclouds_other_pl, labels_pl


# Input point cloud and output the global feature
def calc_ldgcnn_feature(point_cloud, pointclouds_other, is_training, bn_decay=None):
    # B: batch size; N: number of points, C: channels; k: number of nearest neighbors
    # point_cloud: B*N*3
    k = 20
    # ======try to add more features here
    point_cloud = tf.concat(axis=2, values=[point_cloud, pointclouds_other])

    # adj_matrix: B*N*N
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)

    point_cloud = tf.expand_dims(point_cloud, axis=-2)

    # Edge_feature: B*N*k*6
    # The vector in the last dimension represents: (Xc,Yc,Zc, Xck - Xc, Yck-Yc, Yck-zc)
    # (Xc,Yc,Zc) is the central point. (Xck - Xc, Yck-Yc, Yck-zc) is the edge vector.
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    print(f'first edge shape: {edge_feature.shape}')

    # net: B*N*k*64
    # The kernel size of CNN is 1*1, and thus this is a MLP with sharing parameters.
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)

    # net: B*N*1*64
    # Extract the biggest feature from k convolutional edge features.     
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net
    print(f'First EdgeConv layers: {net1.shape}')

    # adj_matrix: B*N*N
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)

    # net: B*N*1*67 
    # Link the Hierarchical features.
    net = tf.concat([point_cloud, net1], axis=-1)

    # edge_feature: B*N*k*134
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    # net: B*N*k*64
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    # net: B*N*1*64
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)

    # net: B*N*1*131
    net = tf.concat([point_cloud, net1, net2], axis=-1)

    # edge_feature: B*N*k*262
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    # net: B*N*k*64
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    # net: B*N*1*64
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)

    # net: B*N*1*195
    net = tf.concat([point_cloud, net1, net2, net3], axis=-1)
    # edge_feature: B*N*k*390
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    # net: B*N*k*128
    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)
    # net: B*N*1*128
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net

    # input: B*N*1*323
    # net: B*N*1*1024
    net = tf_util.conv2d(tf.concat([point_cloud, net1, net2, net3,
                                    net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)
    # net: B*1*1*1024
    net = tf.reduce_max(net, axis=1, keep_dims=True)
    # net: B*1024
    net = tf.squeeze(net)
    return net


def get_model_other(point_cloud, pointclouds_other, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0]  # .value
    layers = {}

    # Extract global feature
    net = calc_ldgcnn_feature(point_cloud, pointclouds_other, is_training, bn_decay)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    layers['global_feature'] = net

    # Fully connected layers: classifier
    # net: B*512
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    layers['fc1'] = net
    # Each element is kept or dropped independently, and the drop rate is 0.5.
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')

    # net: B*256
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    layers['fc2'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')

    # net: B*outclass
    net = tf_util.fully_connected(net, para.outputClassN, activation_fn=None, scope='fc3')
    layers['fc3'] = net
    return net, layers


def get_loss_weight(pred, label, end_points, classweight):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Change the label from an integer to the one_hot vector.
    labels = tf.one_hot(indices=label, depth=para.outputClassN)

    # Calculate the loss based on cross entropy method.
    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    loss = tf.multiply(loss, classweight)  # multiply class weight with loss for each object

    # Calculate the mean loss of a batch input.
    mean_classify_loss = tf.reduce_mean(loss)
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


