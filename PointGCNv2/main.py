import os
import pickle
import time

import numpy as np
import tensorflow as tf
from Parameters import Parameters
from model import model_architecture, trainOneEpoch, evaluateOneEpoch
from read_data import load_data, prepareData
from utils import weight_dict_fc
from sklearn.metrics import confusion_matrix


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
start_time = time.time()

# ===============================Hyper parameters========================
para = Parameters()
print(f'This is experiment: {para.expName}')
# =======================================================================

pointNumber = para.pointNumber
neighborNumber = para.neighborNumber

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
    sess.run(init)
    saver = tf.compat.v1.train.Saver()
    learningRate = para.learningRate

    save_model_path = os.path.join(para.modelDir, para.expName)
    weight_dict = weight_dict_fc(trainLabel, para)
    test_confusionM = []
    test_acc_record = []
    test_mean_acc_record = []

    for epoch in range(para.max_epoch):
        print(f'====================epoch {epoch}====================')
        if epoch % 20 == 0:
            learningRate = learningRate / 2  # 1.7
        learningRate = np.max([learningRate, 10e-6])
        print(f'learningRate:\t{learningRate}')

        inputtrainall = np.dstack((inputTrain, trainsf))

        train_average_loss, train_average_acc, loss_reg_average = trainOneEpoch(inputtrainall, scaledLaplacianTrain,
                                                                                trainLabel, para, sess, trainOperation,
                                                                                weight_dict, learningRate)

        save = saver.save(sess, save_model_path)
        print(f'====================train result====================')
        print(f'average loss:\t{train_average_loss}\n'
              f'l2 loss:\t{loss_reg_average}\n'
              f'accuracy:\t{train_average_acc}\n')
        inputtestall = np.dstack((inputTest, testsf))

        test_average_loss, test_average_acc, test_predict = evaluateOneEpoch(inputtestall, scaledLaplacianTest,
                                                                             testLabel, para, sess, trainOperation)

        # calculate mean class accuracy
        test_predict = np.asarray(test_predict)
        test_predict = test_predict.flatten()
        confusion_mat = confusion_matrix(testLabel[0:len(test_predict)], test_predict)
        normalized_confusion = confusion_mat.astype('float') / confusion_mat.sum(axis=1)
        class_acc = np.diag(normalized_confusion)
        mean_class_acc = np.mean(class_acc)
        print(f'====================test result====================')
        print(f'average loss: \t{test_average_loss}\n'
              f'accuracy: \t{test_average_acc}\n'
              f'average class accuracy: \t{mean_class_acc}')
        print(confusion_mat)
        test_confusionM.append(confusion_mat)
        test_acc_record.append(test_average_acc)
        test_mean_acc_record.append(mean_class_acc)

    # save log
    log_Dir = para.logDir
    fileName = para.expName
    with open(log_Dir + '/' + fileName + '_confusion_mat', 'wb') as handle:
        pickle.dump(test_confusionM, handle)
    with open(log_Dir + '/' + fileName + '_overall_acc_record', 'wb') as handle:
        pickle.dump(test_acc_record, handle)
    with open(log_Dir + '/' + fileName + '_mean_class_acc_record', 'wb') as handle:
        pickle.dump(test_mean_acc_record, handle)

end_time = time.time()
run_time = (end_time - start_time) / 3600
print(f'running time:\t{run_time} hrs')
