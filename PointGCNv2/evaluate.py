import os
import sys
import socket

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import read_data
import model
from Parameters import Parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

para = Parameters(evaluation=True)

print(f'This is experiment: {para.expName}')
if para.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pointNumber = para.pointNumber
neighborNumber = para.neighborNumber

LOG_MODEL = para.modelDir
EVAL = para.evallog

# log file
LOG_FOUT = open(os.path.join(EVAL, f'{para.expName}_testresult.txt'), 'w')
LOG_FOUT.write(str(para.__dict__) + '\n')

HOSTNAME = socket.gethostname()


def log_string(out_str):
    if isinstance(out_str, np.ndarray):
        np.savetxt(LOG_FOUT, out_str, fmt='%.2f')
    else:
        LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.Graph().as_default():  # for define a new graph to put in every element
        # ===============================Build model=============================
        trainOperation = model.model_architecture(para)

        # ================================Load test data===============================
        inputTest, testother, testLabel = read_data.load_data(isEvaluation=True)
        scaledLaplacianTest = read_data.prepareData(None, inputTest, neighborNumber, pointNumber, isEvaluation=True)

        # ===============================Train model ================================
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(para.logDir, 'test'), sess.graph)

        sess.run(init)
        saver = tf.compat.v1.train.Saver()
        save_model_path = os.path.join(para.modelDir, f'{para.expName[:6]}.ckpt')
        saver.restore(sess, save_model_path)
        log_string("Model restored.")

        inputtestall = np.dstack((inputTest, testother))  # add the safety factor
        fout = open(os.path.join(EVAL, 'pred_label.txt'), 'w')
        fout.write('no predict real\n')
        test_average_loss, test_average_acc, test_predict = model.evaluateOneEpoch(inputtestall,
                                                                                   scaledLaplacianTest,
                                                                                   testLabel,
                                                                                   para, sess,
                                                                                   trainOperation,
                                                                                   test_writer)
        # calculate mean class accuracy and log result
        test_predict = np.asarray(test_predict)
        test_predict = test_predict.flatten()
        for i in range(len(test_predict)):
            fout.write('%d %d %d\n' % (i, test_predict[i], testLabel[i]))
        confusion_mat = confusion_matrix(testLabel[0:len(test_predict)], test_predict)
        normalized_confusion = confusion_mat.astype('float') / confusion_mat.sum(axis=1, keepdims=True)
        class_acc = np.diag(normalized_confusion)
        mean_class_acc = np.mean(class_acc)
        log_string('Test result:')
        log_string(f'average loss: {test_average_loss:.3f}')
        log_string(f'accuracy: {test_average_acc:.3f}')
        log_string(f'mean class accuracy: {mean_class_acc:.3f}')
        log_string(confusion_mat)
        for i, name in para.classes.items():
            log_string('%10s:\t%0.3f' % (name, class_acc[i]))


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
