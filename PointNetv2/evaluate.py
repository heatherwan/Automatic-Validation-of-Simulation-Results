import importlib
import os
import socket
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import provider
from Parameters import Parameters

para = Parameters(evaluation=True)
if para.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

BATCH_SIZE = para.batchSize
NUM_CLASSES = para.outputClassN
NUM_POINT = para.pointNumber

MODEL = importlib.import_module(para.model)  # import network module
LOG_MODEL = para.logmodelDir
EVAL = para.evallog

# log file
LOG_FOUT = open(os.path.join(EVAL, f'{para.expName}_testresult.txt'), 'w')
LOG_FOUT.write(str(para.__dict__) + '\n')

HOSTNAME = socket.gethostname()


def log_string(out_str):
    if isinstance(out_str, np.ndarray):
        np.savetxt(LOG_FOUT, out_str, fmt='%3d')
    else:
        LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device(''):
        pointclouds_pl, pointclouds_other_pl, labels_pl = MODEL.placeholder_inputs_other(BATCH_SIZE, NUM_POINT)
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


# since the paras in training is with batch size 32, the evaluation should have the same shape.
# So it is better to have the testing as the multiply of batch size 32
def eval_one_epoch(sess, ops):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(EVAL, 'pred_label.txt'), 'w')
    fout.write('no predict real\n')

    # load data
    current_data, current_other, current_label = provider.loadDataFile_other(para.TEST_FILES)
    current_data = current_data[:, 0:NUM_POINT, :]
    current_other = current_other[:, 0:NUM_POINT]
    current_label = np.squeeze(current_label)
    print(current_data.shape)
    weight_dict = provider.weight_dict_fc(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)
    pred_label = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        batchWeight = provider.weights_calculation(current_label[start_idx:end_idx], weight_dict)

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['pointclouds_other_pl']: current_other[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['weights']: batchWeight}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)  # get the predict class number
        correct_count = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct_count
        total_seen += cur_batch_size
        loss_sum += (loss_val * BATCH_SIZE)

        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)
            fout.write('%d %d %d\n' % (i, pred_val[i - start_idx], l))
        pred_label.extend(pred_val)

    log_string(f'mean loss: {(loss_sum / float(total_seen)):.3f}')
    log_string(f'acc: {(total_correct / float(total_seen)):.3f}')
    avg_class_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    log_string(f'avg class acc: {avg_class_acc:.3f}')
    log_string(confusion_matrix(current_label[:len(pred_label)], pred_label))
    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in para.classes.items():
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    with tf.Graph().as_default():
        start_time = time.time()
        evaluate()
        end_time = time.time()
        run_time = (end_time - start_time) / 60
        log_string(f'running time:\t{run_time} mins')
    LOG_FOUT.close()
