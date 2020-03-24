import importlib
import os
import sys
import socket

import numpy as np
import imageio
import tensorflow as tf

import provider
from Parameters import Parameters
from utils import pc_util

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
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate(num_votes):

    with tf.device(''):
        pointclouds_pl, pointclouds_sf_pl, labels_pl = MODEL.placeholder_inputs_sf(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
        weights = tf.compat.v1.placeholder(tf.float32, [None])

        # simple model
        pred, end_points = MODEL.get_model_sf(pointclouds_pl, pointclouds_sf_pl, is_training_pl)
        if para.weighting_scheme == 'weighted':
            loss = MODEL.get_loss_weight(pred, labels_pl, end_points, weights)
        else:
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
           'pointclouds_sf_pl': pointclouds_sf_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'weights': weights}

    eval_one_epoch(sess, ops, num_votes)


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


# since the paras in training is with batch size 32, the evaluation should have the same shape.
# So it is better to have the testing as the multiply of batch size 32
def eval_one_epoch(sess, ops, num_votes=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(EVAL, 'pred_label.txt'), 'w')
    fout.write('no predict real\n')

    # load data
    current_data, current_sf, current_label = provider.loadDataFile_sf(para.TEST_FILES)
    current_data = current_data[:, 0:NUM_POINT, :]
    current_sf = current_sf[:, 0:NUM_POINT]
    current_label = np.squeeze(current_label)
    print(current_data.shape)
    weight_dict = weight_dict_fc(current_label, para)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        batchWeight = weights_calculation(current_label[start_idx:end_idx], weight_dict)
        # Aggregating BEG
        batch_loss_sum = 0  # sum of losses for the batch
        batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES))  # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES))  # 0/1 for classes
        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                                                vote_idx / float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['pointclouds_sf_pl']: current_sf[start_idx:end_idx, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,
                         ops['weights']: batchWeight}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))

        pred_val = np.argmax(batch_pred_sum, 1)
        # Aggregating END

        correct_count = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct_count
        total_seen += cur_batch_size
        loss_sum += batch_loss_sum

        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)
            fout.write('%d %d %d\n' % (i, pred_val[i - start_idx], l))

            if pred_val[i - start_idx] != l:  # ERROR CASE, DUMP!
                img_filename = 'no_%d_label_%s_pred_%s.jpg' % (i, l, pred_val[i - start_idx])
                img_filename = os.path.join(para.evallog, img_filename)
                output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                imageio.imwrite(img_filename, output_img)
                error_cnt += 1

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in para.classes.items():
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
