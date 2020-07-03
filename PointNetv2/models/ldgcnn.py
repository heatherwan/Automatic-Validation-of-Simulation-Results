"""
The network model of linked dynamic graph CNN.
Reference code: https://github.com/KuangenZhang/ldgcnn
Modified for more features implementation
@author: Wanting Lin

"""
import os
import sys

import tensorflow as tf

from Parameters import Parameters
from utils import tf_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
para = Parameters()


# Add input placeholder
def placeholder_inputs_other(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, para.dim))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, labels_pl


# Input point cloud and output the global feature
def get_model_other(point_cloud, is_training, bn_decay=None):
    """
      B: batch size;
      N: number of points,
      C: channels;
      k: number of nearest neighbors
      point_cloud: B*N*C
    """

    end_points = {}
    minSF = tf.reshape(tf.math.argmin(point_cloud[:, :, 0], axis=1), (-1, 1))
    batch_size = point_cloud.get_shape()[0]  # .value

    # # 1. graph for first EdgeConv B N C=6

    adj_matrix = tf_util.pairwise_distance(point_cloud[:, :, :para.dim])  # B N C=6 => B*N*N
    # adj_matrix = tf_util.pairwise_distance(point_cloud[:, :, 1:4])  # B N C=6 => B*N*N
    nn_idx = tf_util.knn(adj_matrix, k=10)

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    end_points['knn1'] = allSF_dist

    point_cloud = tf.expand_dims(point_cloud[:, :, :para.dim], axis=-2)
    # point_cloud = tf.expand_dims(point_cloud[:, :, 1:4], axis=-2)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=10)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net1 = net

    # # 2. graph for second EdgeConv B N C=64
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=20)

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    end_points['knn2'] = allSF_dist

    # net: B*N*1*6+64=71
    net = tf.concat([point_cloud, net1], axis=-1)

    # edge_feature: B*N*k*142
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=20)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net2 = net

    # 3. graph for third EdgeConv B N C=64
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=30)

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    end_points['knn3'] = allSF_dist

    # net: B*N*1*6+64+64=134
    net = tf.concat([point_cloud, net1, net2], axis=-1)

    # edge_feature: B*N*k*268
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=30)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net3 = net

    # # 4. graph for fourth EdgeConv B N C=64
    # adj_matrix = tf_util.pairwise_distance(net)
    # nn_idx = tf_util.knn(adj_matrix, k=40)
    #
    # # get the distance to minSF of 1024 points
    # allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    # end_points['knn4'] = allSF_dist
    #
    # # net: B*N*1*6+64+64+64=198
    # net = tf.concat([point_cloud, net1, net2, net3], axis=-1)
    #
    # # edge_feature: B*N*k*396
    # edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=40)
    # net = tf_util.conv2d(edge_feature, 128, [1, 1],
    #                      padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training,
    #                      scope='dgcnn4', bn_decay=bn_decay)
    # net = tf.reduce_max(net, axis=-2, keepdims=True)
    # net4 = net

    # input: B*N*1*6+64+64+64+128 = 326  => net: B*N*1*1024
    net = tf_util.conv2d(tf.concat([point_cloud, net1, net2, net3], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)
    # net: B*1*1*1024
    net = tf.reduce_max(net, axis=1, keepdims=True)
    # SF_features = tf.gather(net, indices=minSF, axis=1, batch_dims=1)

    # net: B*1024
    net = tf.squeeze(net)
    # net = tf.squeeze(SF_features)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    print(net.get_shape())
    end_points['global_feature'] = net

    # Fully connected end_points: classifier
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['fc1'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    end_points['fc2'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, para.outputClassN, activation_fn=None, scope='fc3')
    end_points['fc3'] = net
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
