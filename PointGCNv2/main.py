import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from Parameters import Parameters
from model import model_architecture, trainOneEpoch, evaluateOneEpoch
from read_data import load_data, prepareData
from utils import weight_dict_fc

start_time = time.time()

# ===============================Hyper parameters========================
para = Parameters(evaluation=False)
print(f'This is experiment: {para.expName}')
if para.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pointNumber = para.pointNumber
neighborNumber = para.neighborNumber
# ===============================Log setting=============================
LOG_DIR = para.logDir
LOG_MODEL = para.modelDir
LOG_FOUT = open(os.path.join(LOG_DIR, f'{para.expName}.txt'), 'w')
# write parameters in log file
LOG_FOUT.write(str(para.__dict__) + '\n')
# =======================================================================


def log_string(out_str):
    if isinstance(out_str, np.ndarray):
        np.savetxt(LOG_FOUT, out_str, fmt='%.2f')
    else:
        LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


with tf.Graph().as_default():  # for define a new graph to put in every element
    # ===============================Build model=============================
    trainOperation = model_architecture(para)
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # ================================Load data===============================
    inputTrain, trainsf, trainLabel, inputTest, testsf, testLabel = load_data(pointNumber, para.samplingType)
    scaledLaplacianTrain, scaledLaplacianTest = prepareData(inputTrain, inputTest, neighborNumber, pointNumber)

    # ===============================Train model ================================
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()

    train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                                   sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'),
                                                  sess.graph)
    sess.run(init)
    saver = tf.compat.v1.train.Saver()
    learningRate = para.learningRate

    save_model_path = os.path.join(para.modelDir, para.expName[:6], '.ckpt')
    weight_dict = weight_dict_fc(trainLabel, para)
    test_confusionM = []
    test_acc_record = []
    test_mean_acc_record = []

    for epoch in range(para.max_epoch):
        log_string(f'====================epoch {epoch}====================')
        if epoch % 20 == 0:
            learningRate = learningRate / 2  # 1.7
        learningRate = np.max([learningRate, 10e-6])
        log_string(f'learningRate:\t{learningRate}')

        inputtrainall = np.dstack((inputTrain, trainsf))

        train_average_loss, train_average_acc, loss_reg_average = trainOneEpoch(inputtrainall, scaledLaplacianTrain,
                                                                                trainLabel, para, sess, trainOperation,
                                                                                weight_dict, learningRate, train_writer)
        log_string('Train result:')
        log_string(f'mean loss: {loss_reg_average}')
        log_string(f'accuracy: {train_average_acc}')

        save = saver.save(sess, save_model_path)

        inputtestall = np.dstack((inputTest, testsf))  # add the safety factor
        test_average_loss, test_average_acc, test_predict = evaluateOneEpoch(inputtestall, scaledLaplacianTest,
                                                                             testLabel, para, sess, trainOperation,
                                                                             test_writer)
        # calculate mean class accuracy and log result
        test_predict = np.asarray(test_predict)
        test_predict = test_predict.flatten()
        confusion_mat = confusion_matrix(testLabel[0:len(test_predict)], test_predict)
        normalized_confusion = confusion_mat.astype('float') / confusion_mat.sum(axis=1, keepdims=True)
        class_acc = np.diag(normalized_confusion)
        mean_class_acc = np.mean(class_acc)
        log_string('Test result:')
        log_string(f'average loss: {test_average_loss}')
        log_string(f'accuracy: {test_average_acc}')
        log_string(f'mean class accuracy: {mean_class_acc}')
        log_string(normalized_confusion)

end_time = time.time()
run_time = (end_time - start_time) / 60
log_string(f'running time:\t{run_time} min')
LOG_FOUT.close()
