"""
The classifier of linked dynamic graph CNN.
@author: Kuangen Zhang

"""
import tensorflow as tf
import numpy as np
import sys
import os
from utils import tf_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def get_model(feature, is_training, bn_decay=None):
    # Fully connected layers: classifier
    layers = {}
    feature = tf.squeeze(feature)
    layer_name = 'ft_'

    # B: batch size; C: channels;
    # feature: B*C
    # net: B*512
    net = tf_util.fully_connected(feature, 512, bn=True, is_training=is_training,
                                  scope=layer_name + 'fc2', bn_decay=bn_decay,
                                  activation_fn=tf.nn.relu)
    layers[layer_name + 'fc2'] = net

    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope=layer_name + 'dp2')

    # net: B*256
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope=layer_name + 'fc3', bn_decay=bn_decay,
                                  activation_fn=tf.nn.relu)
    layers[layer_name + 'fc3'] = net

    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope=layer_name + 'dp3')
    # net: B*40
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc4')
    layers[layer_name + 'fc4'] = net

    return net, layers


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
      label: B, """
    # Change the label from an integer to the one_hot vector.
    labels = tf.one_hot(indices=label, depth=40)
    # Calculate the loss based on cross entropy method.
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    # Calculate the mean loss of a batch input.
    classify_loss = tf.reduce_mean(loss)
    return classify_loss


