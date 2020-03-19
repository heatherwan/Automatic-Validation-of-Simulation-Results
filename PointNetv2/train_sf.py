import provider
import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from sklearn.metrics import confusion_matrix
from Parameters import Parameters
import collections

para = Parameters()

os.environ["CUDA_VISIBLE_DEVICES"] = para.gpu

BATCH_SIZE = para.batchSize
testBatch = para.testBatchSize
NUM_CLASSES = para.outputClassN
NUM_POINT = para.pointNumber
MAX_EPOCH = para.max_epoch
BASE_LEARNING_RATE = para.learningRate
MOMENTUM = para.momentum
OPTIMIZER = para.optimizer
DECAY_STEP = para.decay_step
DECAY_RATE = para.decay_rate

MODEL = importlib.import_module(para.model)  # import network module
LOG_DIR = para.log_dir
LOG_MODEL = para.logmodelDir
# log file
LOG_FOUT = open(os.path.join(LOG_DIR, f'{para.expName}.txt'), 'w')
# write parameters in log file

LOG_FOUT.write(str(para.__dict__) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
TRAIN_FILES = para.TRAIN_FILES
TEST_FILES = para.TEST_FILES


def log_string(out_str):
    if isinstance(out_str, np.ndarray):
        np.savetxt(LOG_FOUT, out_str, fmt='%3d')
    else:
        LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device(''):
            pointclouds_pl, pointclouds_sf_pl, labels_pl = MODEL.placeholder_inputs_sf(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            weights = tf.compat.v1.placeholder(tf.float32, [None])
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model_sf(pointclouds_pl, pointclouds_sf_pl, is_training_pl, bn_decay=bn_decay)
            if para.weighting_scheme == 'weighted':
                loss = MODEL.get_loss_weight(pred, labels_pl, end_points, weights)
            else:
                loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(input=pred, axis=1), tf.cast(labels_pl, dtype=tf.int64))
            accuracy = tf.reduce_sum(input_tensor=tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()

        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)

        # Init variables
        init = tf.compat.v1.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_sf_pl': pointclouds_sf_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'weights': weights}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_MODEL, f"{para.expName}.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def weight_dict_fc(trainLabel, para):
    from sklearn.preprocessing import label_binarize
    y_total = label_binarize(trainLabel, classes=[i for i in range(para.outputClassN)])
    class_distribution_class = np.sum(y_total, axis=0)  # get count for each class
    class_distribution_class = [float(i) for i in class_distribution_class]
    class_distribution_class = class_distribution_class / np.sum(class_distribution_class)  # get ratio for each class
    inverse_dist = 1 / class_distribution_class
    norm_inv_dist = inverse_dist / np.sum(inverse_dist)
    weights = norm_inv_dist * para.weight_scaler + 1  # scalar should be reconsider
    weight_dict = dict()
    for classID, value in enumerate(weights):
        weight_dict.update({classID: value})
    return weight_dict


def weights_calculation(batch_labels, weight_dict):
    weights = []
    # batch_labels = np.argmax(batch_labels, axis=1)

    for i in batch_labels:
        weights.append(weight_dict[i])
    return weights


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    current_data, current_sf, current_label = provider.loadDataFile_sf(TRAIN_FILES)
    current_data = current_data[:, 0:NUM_POINT, :]
    current_sf = current_sf[:, 0:NUM_POINT]
    current_data, current_sf, current_label, _ = provider.shuffle_data_sf(current_data, current_sf,
                                                                          np.squeeze(current_label))
    current_label = np.squeeze(current_label)
    # ===================implement weight here ==================
    weight_dict = weight_dict_fc(current_label, para)

    # ===========================================================
    file_size = current_data.shape[0]
    # counter = collections.Counter(current_label)
    # print(counter)
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    total_pred = []
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batchWeight = weights_calculation(current_label[start_idx:end_idx], weight_dict)

        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        jittered_data = provider.jitter_point_cloud(rotated_data)
        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['pointclouds_sf_pl']: current_sf[start_idx:end_idx, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['weights']: batchWeight}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        total_pred.extend(pred_val)
    log_string('Train result:')
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    log_string(confusion_matrix(current_label[:len(total_pred)], total_pred))


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_pred = []

    current_data, current_sf, current_label = provider.loadDataFile_sf(TEST_FILES)
    current_data = current_data[:, 0:NUM_POINT, :]
    current_sf = current_sf[:, 0:NUM_POINT]
    current_label = np.squeeze(current_label)
    weight_dict = weight_dict_fc(current_label, para)
    file_size = current_data.shape[0]
    num_batches = file_size // testBatch

    for batch_idx in range(num_batches):
        start_idx = batch_idx * testBatch
        end_idx = (batch_idx + 1) * testBatch
        batchWeight = weights_calculation(current_label[start_idx:end_idx], weight_dict)
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['pointclouds_sf_pl']: current_sf[start_idx:end_idx, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['weights']: batchWeight}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += testBatch
        loss_sum += (loss_val * testBatch)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)
        total_pred.extend(pred_val)

    log_string('Test result:')
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    log_string(confusion_matrix(current_label[:len(total_pred)], total_pred))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
