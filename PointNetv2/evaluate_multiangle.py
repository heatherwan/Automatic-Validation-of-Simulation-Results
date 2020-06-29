import importlib
import os
import socket
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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
LOG_FOUT = open(os.path.join(EVAL, f'{para.expName}_testresult_vote{para.num_votes}.txt'), 'w')
LOG_FOUT.write(str(para.__dict__) + '\n')

HOSTNAME = socket.gethostname()

# get dataset
testDataset = DatasetHDF5(para.TEST_FILES, batch_size=para.testBatchSize,
                          npoints=para.pointNumber, dim=para.dim, shuffle=False, train=False)


def log_string(out_str):
    if isinstance(out_str, np.ndarray):
        np.savetxt(LOG_FOUT, out_str, fmt='%3d')
    else:
        LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device(''):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs_other(para.testBatchSize, para.pointNumber)
        is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model_other(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
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
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'knn': end_points}

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((para.testBatchSize, para.pointNumber, para.dim))
    cur_batch_label = np.zeros(para.testBatchSize, dtype=np.int32)

    # set variable for statistics
    total_correct = 0
    total_seen = 0
    pred_label = []
    batch_idx = 0

    fout = open(os.path.join(EVAL, f'{para.expName[:6]}_all_pred_label_vote{para.num_votes}.txt'), 'w')
    fout.write('  no\tpred\treal\tGood\tContact\tRadius\tHole\n')
    # fout2 = open(os.path.join(EVAL, f'{para.expName[:6]}_wrong_pred_prob.txt'), 'w')
    # fout2.write('  no\tpred\treal\tGood\tContact\tRadius\tHole\n')
    # all_knn_idx = {}
    while testDataset.has_next_batch():
        batch_data, batch_label = testDataset.next_batch(augment=False)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data[:, :, :para.dim]
        cur_batch_label[0:bsize] = batch_label

        all_pred_prob = np.zeros((cur_batch_data.shape[0], para.outputClassN))

        for vote_idx in range(para.num_votes):
            cur_batch_data[:, :, 1:4] = provider.rotate_point_cloud_by_angle(cur_batch_data[:, :, 1:4],
                                                                            vote_idx / float(
                                                                                para.num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}

            loss_val, logits, knn_idx = sess.run([ops['loss'], ops['pred'], ops['knn']], feed_dict=feed_dict)
            pred_prob = np.exp(logits) / np.sum(np.exp(logits), axis=1).reshape(para.batchSize, 1)
            all_pred_prob = np.add(all_pred_prob, pred_prob)
            # record result for a batch
            pred_val = np.argmax(logits, 1)  # get the predict class number
            # for i in range(bsize):
            #     fout.write(f'{batch_idx * para.testBatchSize + i:^5d}\t{pred_val[i]:^5d}\t{batch_label[i]:^5d}\t')
            #     for num in range(para.outputClassN):
            #         fout.write(f'{pred_prob[i][num]:.3f}\t')
            #     fout.write('\n')

        # mean pred and count the class accuracy
        mean_pred_prob = all_pred_prob/para.num_votes
        mean_pred_val = np.argmax(mean_pred_prob, 1)  # get the predict class number

        # log average result
        for i in range(bsize):
            fout.write(f'{batch_idx * para.testBatchSize + i:^5d}\t{mean_pred_val[i]:^5d}\t{batch_label[i]:^5d}\t')
            for num in range(para.outputClassN):
                fout.write(f'{mean_pred_prob[i][num]:.3f}\t')
            fout.write('\n')

        pred_label.extend(mean_pred_val[0:bsize])
        batch_idx += 1
        correct_count = np.sum(mean_pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct_count
        total_seen += bsize

    log_string('Test result:')
    log_string(f'acc: {(total_correct / float(total_seen)):.3f}')
    log_string(f'avg class acc: \n')
    if len(np.unique(testDataset.current_label)) == 4:
        log_string(classification_report(testDataset.current_label[:len(pred_label)], pred_label,
                                         target_names=['Good', 'Contact', 'Radius', 'Hole'], digits=3))
    elif len(np.unique(testDataset.current_label)) == 3:
        log_string(classification_report(testDataset.current_label[:len(pred_label)], pred_label,
                                         target_names=['Good', 'Contact', 'Hole'], digits=3))

    log_string(confusion_matrix(testDataset.current_label[:len(pred_label)], pred_label))

    # if para.model == "dgcnn" or para.model == 'ldgcnn' or para.model == 'ldgcnn_2layer':
    #     for k, v in all_knn_idx.items():
    #         v.tofile(f'evallog/{para.expName[:6]}_{k}.txt', sep=" ", format="%.3f")


if __name__ == '__main__':
    with tf.Graph().as_default():
        start_time = time.time()
        evaluate()
        end_time = time.time()
        run_time = (end_time - start_time) / 60
        log_string(f'running time:\t{run_time} mins')
    LOG_FOUT.close()
