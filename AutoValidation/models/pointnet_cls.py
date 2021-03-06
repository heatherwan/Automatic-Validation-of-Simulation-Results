import numpy as np
import tensorflow as tf

from Parameters import Parameters
from models.transform_nets import input_transform_net, feature_transform_net
from utils import tf_util

para = Parameters()


def placeholder_inputs_other(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, para.dim))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, labels_pl


def get_model_other(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3 and BxNx1, output Bx4 """
    batch_size = point_cloud.get_shape()[0]  # .value
    num_point = point_cloud.get_shape()[1]  # .value 
    end_points = {}

    # transform net for input x,y,z
    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud[:, :, 1:4], is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud[:, :, 1:4], transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    # First MLP layers
    net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    # transform net for the features
    with tf.compat.v1.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    # point_feat = tf.expand_dims(net_transformed, [2])

    # add the additional features to the second MLP layers
    # point_cloud_SF = tf.expand_dims(point_cloud[:, :, 0:1], [2])
    concat_other = tf.concat(axis=2, values=[point_cloud[:, :, 0:1], net_transformed, point_cloud[:, :, 4:para.dim]])
    concat_other = tf.expand_dims(concat_other, [2])

    # second MLP layers
    net = tf_util.conv2d(concat_other, 64, [1, 1],  # 64 is the output #neuron
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 4, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred:   Batchsize*NUM_CLASSES,   probabilities for each class for B objects
        label:  Batchsize,               real label for B object
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    mean_classify_loss = tf.reduce_mean(input_tensor=loss)
    tf.compat.v1.summary.scalar('classify loss', mean_classify_loss)

    # Enforce the transformation as orthogonal matrix,
    transform = end_points['transform']  # BxKxK
    K = transform.get_shape()[1]  # .value
    mat_diff = tf.matmul(transform, tf.transpose(a=transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)  # mat_diff = I-AA.T transform net close to orthogonal  AA.T == I
    tf.compat.v1.summary.scalar('mat loss', mat_diff_loss)

    return mean_classify_loss + mat_diff_loss * reg_weight


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


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model_other(inputs, tf.constant(True))
        print(outputs)
