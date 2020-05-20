import importlib
import os
import socket
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import datetime

import provider
from Parameters import Parameters
from Dataset_hdf5 import DatasetHDF5

# ===============get basic folder=====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
para = Parameters()

# log
MODEL = importlib.import_module(para.model)  # import network module
if para.model == 'ldgcnn':
    MODEL_CLS = importlib.import_module(para.class_model)
LOG_DIR = para.logDir
LOG_MODEL = para.logmodelDir
LOG_FOUT = open(os.path.join(LOG_DIR, f'{para.expName}.txt'), 'w')
LOG_FOUT.write(str(para.__dict__) + '\n')

# set parameters
if para.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(para.decay_step)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# get dataset
trainDataset = DatasetHDF5(para.TRAIN_FILES, batch_size=para.batchSize,
                           npoints=para.pointNumber, dim=para.dim, shuffle=True)
testDataset = DatasetHDF5(para.TEST_FILES, batch_size=para.testBatchSize,
                          npoints=para.pointNumber, dim=para.dim, shuffle=False)


def log_string(out_str):
    # for confusion matrix
    if isinstance(out_str, np.ndarray):
        np.savetxt(LOG_FOUT, out_str, fmt='%3d')
    else:
        LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
        para.learningRate,  # Base learning rate.
        batch * para.batchSize,  # Current index into the dataset.
        para.decay_step,  # Decay step.
        para.decay_rate,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
        BN_INIT_DECAY,
        batch * para.batchSize,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device(''):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs_other(para.batchSize, para.pointNumber)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            weights = tf.compat.v1.placeholder(tf.float32, [None])
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model_other(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            para_num = MODEL.get_para_num()
            print(f'Total parameters number is {para_num}')
            LOG_FOUT.write(str(para_num) + '\n')

            loss = MODEL.get_loss_weight(pred, labels_pl, end_points, weights)

            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(input=pred, axis=1), tf.cast(labels_pl, dtype=tf.int64))
            accuracy = tf.reduce_sum(input_tensor=tf.cast(correct, tf.float32)) / float(para.batchSize)
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if para.optimizer == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=para.momentum)
            elif para.optimizer == 'adam':
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
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, para.expName[:6] + 'train'),
                                                       sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, para.expName[:6] + 'test'),
                                                      sess.graph)
        # Init variables
        init = tf.compat.v1.global_variables_initializer()

        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'weights': weights,
               'knn': end_points}

        min_loss = np.inf
        for epoch in range(para.max_epoch):
            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            loss = train_one_epoch(sess, ops, train_writer)
            trainDataset.reset()
            if epoch % 10 == 0:  # test every 10 epoch
                eval_one_epoch(sess, ops, test_writer)
                testDataset.reset()

            if loss < min_loss:  # save the min loss model
                save_path = saver.save(sess, os.path.join(LOG_MODEL, f"{para.expName[:6]}.ckpt"))
                log_string("Model saved in file: %s" % save_path)
                min_loss = loss

        # Save the extracted global feature
        if para.model == 'ldgcnn':
            save_global_feature(sess, ops, saver, end_points)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((para.batchSize, para.pointNumber, para.dim))
    cur_batch_label = np.zeros(para.batchSize, dtype=np.int32)

    # set variable for statistics
    total_correct = 0
    total_seen = 0
    total_pred = []
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(para.outputClassN)]
    total_correct_class = [0 for _ in range(para.outputClassN)]

    while trainDataset.has_next_batch():
        batch_data, batch_label = trainDataset.next_batch(augment=True)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batchWeight = provider.weights_calculation(cur_batch_label, trainDataset.weight_dict)

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,
                     ops['weights']: batchWeight}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)

        train_writer.add_summary(summary, step)  # tensorboard
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val * bsize
        total_pred.extend(pred_val[0:bsize])
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string('Train result:')
    log_string(f'mean loss: {loss_sum / float(total_seen):.3f}')
    log_string(f'accuracy: {total_correct / float(total_seen):.3f}')
    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    avg_class_acc = np.mean(class_accuracies)
    log_string(f'avg class acc: {avg_class_acc:.3f}')
    for i, name in para.classes.items():
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    log_string(confusion_matrix(trainDataset.current_label[:len(total_pred)], total_pred))
    return loss_sum / float(total_seen)


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((para.testBatchSize, para.pointNumber, para.dim))
    cur_batch_label = np.zeros(para.testBatchSize, dtype=np.int32)

    # set variable for statistics
    total_correct = 0
    total_seen = 0
    pred_label = []
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(para.outputClassN)]
    total_correct_class = [0 for _ in range(para.outputClassN)]
    while testDataset.has_next_batch():
        batch_data, batch_label = testDataset.next_batch(augment=False)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        batchWeight = provider.weights_calculation(cur_batch_label, testDataset.weight_dict)

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,
                     ops['weights']: batchWeight}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']],
                                                     feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val * bsize
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
        pred_label.extend(pred_val[0:bsize])

    log_string('Test result:')
    log_string(f'mean loss: {(loss_sum / float(total_seen)):.3f}')
    log_string(f'acc: {(total_correct / float(total_seen)):.3f}')
    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    avg_class_acc = np.mean(class_accuracies)
    log_string(f'avg class acc: {avg_class_acc:.3f}')
    for i, name in para.classes.items():
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    log_string(confusion_matrix(testDataset.current_label[:len(pred_label)], pred_label))


def save_global_feature(sess, ops, saver, layers):
    feature_name = 'global_feature'
    datasets = [trainDataset, testDataset]
    # Restore variables that achieves the best validation accuracy from the disk.
    saver.restore(sess, os.path.join(LOG_MODEL, f"{para.expName[:6]}.ckpt"))
    log_string("Model restored.")
    is_training = False

    # Extract the features from training set and validation set.
    for r in range(2):
        dataset = datasets[r]
        global_feature_vec = np.array([])
        label_vec = np.array([])
        # Make sure batch data is of same size
        cur_batch_data = np.zeros((para.batchSize, para.pointNumber, para.dim))
        cur_batch_label = np.zeros(para.batchSize, dtype=np.int32)

        while dataset.has_next_batch():
            batch_data, batch_label = dataset.next_batch(augment=False)
            bsize = batch_data.shape[0]
            cur_batch_data[0:bsize, ...] = batch_data
            cur_batch_label[0:bsize] = batch_label
            # Input the point cloud and labels to the graph.
            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}

            # Extract the global features from the input batch data.
            global_feature = np.squeeze(layers[feature_name].eval(feed_dict=feed_dict, session=sess))

            if label_vec.shape[0] == 0:
                global_feature_vec = global_feature[0:bsize, ...]
                label_vec = batch_label
            else:
                global_feature_vec = np.concatenate([global_feature_vec, global_feature[0:bsize, ...]])
                label_vec = np.concatenate([label_vec, batch_label])

        # Save all global features to the disk.
        dataset.set_feature(global_feature_vec, label_vec)


def train_classifier():
    with tf.Graph().as_default():
        with tf.device(''):
            pointclouds_feature_pl, labels_pl = MODEL_CLS.placeholder_inputs_feature(para.batchSize)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, layers = MODEL_CLS.get_model(pointclouds_feature_pl, is_training_pl, bn_decay=bn_decay)

            loss = MODEL_CLS.get_loss(pred, labels_pl)
            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.cast(labels_pl, dtype=tf.int64))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(para.batchSize)
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            # We change the optimier to momentum. Because the momentum can
            # find better parameters
            if para.class_optimizer == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=para.momentum)
            elif para.class_optimizer == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            elif para.class_optimizer == 'SGD':
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
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
        train_writer_cls = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, para.expName[:6] + 'train_cls'),
                                                           sess.graph)
        test_writer_cls = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, para.expName[:6] + 'test_cls'),
                                                          sess.graph)

        # We retrain the classifier three times to see the stable accuracy.
        # It takes much less time to retrain the classifier than to
        # train the whole network.
        # idx_num = 3
        #
        # for idx in range(idx_num):

        # Init variables
        init = tf.compat.v1.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_feature_pl': pointclouds_feature_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        min_loss = np.inf
        for epoch in range(para.class_max_epoch):
            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            loss = train_classifier_one_epoch(sess, ops, train_writer_cls)
            trainDataset.reset_feature()
            eval_classifier_one_epoch(sess, ops, test_writer_cls)
            testDataset.reset_feature()

            if loss < min_loss:  # save the min loss model
                save_path = saver.save(sess, os.path.join(LOG_MODEL, f"{para.expName[:6]}_cls.ckpt"))
                log_string("Model saved in file: %s" % save_path)
                min_loss = loss


def train_classifier_one_epoch(sess, ops, train_writer_cls):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_feature = np.zeros((para.batchSize, para.class_feature))
    cur_batch_label = np.zeros(para.batchSize, dtype=np.int32)

    # set variable for statistics
    total_correct = 0
    total_seen = 0
    total_pred = []
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(para.outputClassN)]
    total_correct_class = [0 for _ in range(para.outputClassN)]

    while trainDataset.has_next_feature():
        # Shuffle train files
        batch_feature, batch_label = trainDataset.next_feature()
        bsize = batch_feature.shape[0]
        cur_batch_feature[0:bsize, ...] = batch_feature
        cur_batch_label[0:bsize] = batch_label

        # Input the features and labels to the graph.
        feed_dict = {ops['pointclouds_feature_pl']: cur_batch_feature,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}

        # Calculate the loss and classification scores.
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)

        train_writer_cls.add_summary(summary, step)  # tensorboard
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val * bsize
        total_pred.extend(pred_val[0:bsize])
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string('Train result:')
    log_string(f'mean loss: {loss_sum / float(total_seen):.3f}')
    log_string(f'accuracy: {total_correct / float(total_seen):.3f}')
    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    avg_class_acc = np.mean(class_accuracies)
    log_string(f'avg class acc: {avg_class_acc:.3f}')
    for i, name in para.classes.items():
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    log_string(confusion_matrix(trainDataset.current_feature_label[:len(total_pred)], total_pred))
    return loss_sum / float(total_seen)


def eval_classifier_one_epoch(sess, ops, test_writer_cls):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_feature = np.zeros((para.batchSize, para.class_feature))
    cur_batch_label = np.zeros(para.batchSize, dtype=np.int32)

    # set variable for statistics
    total_correct = 0
    total_seen = 0
    total_pred = []
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(para.outputClassN)]
    total_correct_class = [0 for _ in range(para.outputClassN)]

    while testDataset.has_next_feature():
        # Shuffle train files
        batch_feature, batch_label = testDataset.next_feature()
        bsize = batch_feature.shape[0]
        cur_batch_feature[0:bsize, ...] = batch_feature
        cur_batch_label[0:bsize] = batch_label

        # Input the features and labels to the graph.
        feed_dict = {ops['pointclouds_feature_pl']: cur_batch_feature,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}

        # Calculate the loss and classification scores.
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)

        test_writer_cls.add_summary(summary, step)  # tensorboard
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val * bsize
        total_pred.extend(pred_val[0:bsize])
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string('Test result:')
    log_string(f'mean loss: {loss_sum / float(total_seen):.3f}')
    log_string(f'accuracy: {total_correct / float(total_seen):.3f}')
    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    avg_class_acc = np.mean(class_accuracies)
    log_string(f'avg class acc: {avg_class_acc:.3f}')
    for i, name in para.classes.items():
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    log_string(confusion_matrix(testDataset.current_feature_label[:len(total_pred)], total_pred))
    return loss_sum / float(total_seen)


if __name__ == "__main__":
    start_time = time.time()
    if para.model == 'ldgcnn':
        train()
        train_classifier()
    else:
        train()
    end_time = time.time()
    run_time = (end_time - start_time) / 60
    log_string(f'running time:\t{run_time} mins')

    LOG_FOUT.close()
