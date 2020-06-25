"""
The network model of dynamic graph CNN.
Reference code: https://github.com/WangYueFt/dgcnn
Modified for more features implementation
@author: Wanting Lin

"""
import tensorflow as tf

from utils import tf_util
from models.transform_nets import input_transform_net_dgcnn
from Parameters import Parameters

para = Parameters()


def placeholder_inputs_other(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, para.dim))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, labels_pl


def get_model_other(point_cloud, is_training, bn_decay=None):
    """ Classification DGCNN, input is BxNxC, output BxCls
        B batch size
        N number of points per pointcloud
        C input channel: eg. x,y,z,SF,distance, minSF...
        Cls output class number
    """
    batch_size = point_cloud.get_shape()[0]  # .value
    end_points = {}

    # get MinSF index
    minSF = tf.reshape(tf.math.argmin(point_cloud[:, :, 0], axis=1), (-1, 1))

    # # 1. graph for transform net with only x,y,z
    adj_matrix = tf_util.pairwise_distance(point_cloud[:, :, 1:])  # B N C=3 => B N N
    nn_idx = tf_util.knn(adj_matrix, k=para.k)
    edge_feature = tf_util.get_edge_feature(point_cloud[:, :, 1:], nn_idx=nn_idx, k=para.k)
    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net_dgcnn(edge_feature, is_training, bn_decay, K=3)
    point_cloud_transform = tf.matmul(point_cloud[:, :, 1:], transform)

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)  # B N 1
    end_points['knn1'] = allSF_dist

    # # 2. graph for first EdgeConv with transform(x,y,z), SF, distance, minSF
    point_cloud_all = tf.concat(axis=2, values=[point_cloud[:, :, 0],
                                                point_cloud_transform,
                                                point_cloud[:, :, 4:para.dim]])

    adj_matrix = tf_util.pairwise_distance(point_cloud_all)  # B N C=6
    nn_idx = tf_util.knn(adj_matrix, k=para.k)
    edge_feature = tf_util.get_edge_feature(point_cloud_all, nn_idx=nn_idx, k=para.k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)
    net = tf.reduce_max(input_tensor=net, axis=-2, keepdims=True)
    net1 = net

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    end_points['knn2'] = allSF_dist

    # # 3. graph for second EdgeConv with C = 64
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=para.k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=para.k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    net = tf.reduce_max(input_tensor=net, axis=-2, keepdims=True)
    net2 = net

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    end_points['knn3'] = allSF_dist

    # # 4. graph for third EdgeConv with C = 64
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=para.k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=para.k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    net = tf.reduce_max(input_tensor=net, axis=-2, keepdims=True)
    net3 = net

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    end_points['knn4'] = allSF_dist

    # # 5. graph for fourth EdgeConv with C = 64
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=para.k)

    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=para.k)
    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)
    net = tf.reduce_max(input_tensor=net, axis=-2, keepdims=True)
    net4 = net

    # get the distance to minSF of 1024 points
    allSF_dist = tf.gather(adj_matrix, indices=minSF, axis=2, batch_dims=1)
    end_points['knn5'] = allSF_dist

    # # 6. MLP for all concatenate features 64+64+64+128
    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)
    net = tf.reduce_max(input_tensor=net, axis=1, keepdims=True)  # maxpooling B N C=1024 => B 1 1024

    # # 7. MLP on global point cloud vector B 1 1024
    net = tf.reshape(net, [batch_size, -1])
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
    if para.outputClassN == 4:
        coarse_label = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(labels),
                                                                 tf.constant([0, 1, 1, 1]), num_segments=2))
    else:
        coarse_label = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(labels),
                                                                 tf.constant([0, 1, 1]), num_segments=2))
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
