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
para = Parameters(evaluation=True)
if para.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# set parameters
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(para.decay_step)
BN_DECAY_CLIP = 0.99

MODEL = importlib.import_module(para.model)  # import network module
LOG_MODEL = para.logmodelDir
EVAL = para.evallog

# log file
LOG_FOUT = open(os.path.join(EVAL, f'{para.expName}_testresult.txt'), 'w')
LOG_FOUT.write(str(para.__dict__) + '\n')

HOSTNAME = socket.gethostname()

# get dataset
testDataset = DatasetHDF5(para.TEST_FILES, batch_size=para.testBatchSize, npoints=para.pointNumber, dim=para.dim, shuffle=True)


def log_string(out_str):
    if isinstance(out_str, np.ndarray):
        np.savetxt(LOG_FOUT, out_str, fmt='%3d')
    else:
        LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device(''):
        pointclouds_pl, pointclouds_other_pl, labels_pl = MODEL.placeholder_inputs_other(para.testBatchSize,
                                                                                         para.pointNumber)
        is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
        weights = tf.compat.v1.placeholder(tf.float32, [None])

        # simple model
        pred, end_points = MODEL.get_model_other(pointclouds_pl, pointclouds_other_pl, is_training_pl)
        loss = MODEL.get_loss_weight(pred, labels_pl, end_points, weights)
        tf.compat.v1.summary.scalar('loss', loss)

        # Add ops to save and restore all the variables.
        saver = tf.compat.v1.train.Saver()

    # Create a session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, f"{LOG_MODEL}/{para.expName[:6]}.ckpt")
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'pointclouds_other_pl': pointclouds_other_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'weights': weights}

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((para.testBatchSize, para.pointNumber, 3))
    cur_batch_other = np.zeros((para.testBatchSize, para.pointNumber, testDataset.num_channel() - 3))
    cur_batch_label = np.zeros(para.testBatchSize, dtype=np.int32)

    # set variable for statistics
    total_correct = 0
    total_seen = 0
    pred_label = []
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(para.outputClassN)]
    total_correct_class = [0 for _ in range(para.outputClassN)]

    fout = open(os.path.join(EVAL, 'all_pred_label.txt'), 'w')
    fout.write('  no\tpred\treal\tGood\tContact\tRadius\tHole\n')
    fout2 = open(os.path.join(EVAL, 'wrong_pred_prob.txt'), 'w')
    fout2.write('  no\tpred\treal\tGood\tContact\tRadius\tHole\n')

    while testDataset.has_next_batch():
        batch_data, batch_other, batch_label = testDataset.next_batch(augment=False)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_other[0:bsize, ...] = batch_other
        cur_batch_label[0:bsize] = batch_label
        batchWeight = provider.weights_calculation(cur_batch_label, testDataset.weight_dict)

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['pointclouds_other_pl']: cur_batch_other,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,
                     ops['weights']: batchWeight}

        loss_val, pred_prob = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

        pred_prob2 = np.exp(pred_prob) / np.sum(np.exp(pred_prob), axis=1).reshape(para.batchSize, 1)
        pred_val = np.argmax(pred_prob, 1)  # get the predict class number
        correct_count = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct_count
        total_seen += bsize
        loss_sum += (loss_val * bsize)
        batch_idx += 1
        for i in range(bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
            fout.write(f'{i:^5d}\t{pred_val[i]:^5d}\t{l:^5d}\t'
                       f'{pred_prob2[i][0]:.3f}\t{pred_prob2[i][1]:.3f}\t'
                       f'{pred_prob2[i][2]:.3f}\t{pred_prob2[i][3]:.3f}\n')
            if pred_val[i] != l:
                fout2.write(f'{i:^5d}\t{pred_val[i]:^5d}\t{l:^5d}\t'
                            f'{pred_prob2[i][0]:.3f}\t{pred_prob2[i][1]:.3f}\t'
                            f'{pred_prob2[i][2]:.3f}\t{pred_prob2[i][3]:.3f}\n')
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


if __name__ == '__main__':
    with tf.Graph().as_default():
        start_time = time.time()
        evaluate()
        end_time = time.time()
        run_time = (end_time - start_time) / 60
        log_string(f'running time:\t{run_time} mins')
    LOG_FOUT.close()
