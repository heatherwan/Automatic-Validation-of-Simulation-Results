import tensorflow as tf
import numpy as np
from layers import gcnLayer, globalPooling, fullyConnected
from utils import get_mini_batch, add_noise, weights_calculation, uniform_weight
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
import sys


# ===========================Hyper parameter=====================
def model_architecture(para):
    inputPC = tf.compat.v1.placeholder(tf.float32, [None, para.pointNumber, para.dim])
    inputGraph = tf.compat.v1.placeholder(tf.float32, [None, para.pointNumber * para.pointNumber])
    outputLabel = tf.compat.v1.placeholder(tf.float32, [None, para.outputClassN])

    scaledLaplacian = tf.reshape(inputGraph, [-1, para.pointNumber, para.pointNumber])

    weights = tf.compat.v1.placeholder(tf.float32, [None])
    lr = tf.compat.v1.placeholder(tf.float32)
    keep_prob_1 = tf.compat.v1.placeholder(tf.float32)
    keep_prob_2 = tf.compat.v1.placeholder(tf.float32)

    # gcn layer 1
    gcn_1 = gcnLayer(inputPC, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.dim,
                     outputFeatureN=para.gcn_1_filter_n,
                     chebyshev_order=para.chebyshev_1_Order)
    gcn_1_output = tf.nn.dropout(gcn_1, rate=1 - (keep_prob_1))
    gcn_1_pooling = globalPooling(gcn_1_output, featureNumber=para.gcn_1_filter_n)
    print("The output of the first gcn layer is {}".format(gcn_1_pooling))
    print(gcn_1_pooling)

    # gcn_layer_2

    gcn_2 = gcnLayer(gcn_1_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_1_filter_n,
                     outputFeatureN=para.gcn_2_filter_n,
                     chebyshev_order=para.chebyshev_2_Order)
    gcn_2_output = tf.nn.dropout(gcn_2, rate=1 - keep_prob_1)
    gcn_2_pooling = globalPooling(gcn_2_output, featureNumber=para.gcn_2_filter_n)
    print("The output of the second gcn layer is {}".format(gcn_2_pooling))

    # concatenate global features
    globalFeatures = tf.concat([gcn_1_pooling, gcn_2_pooling], axis=1)
    globalFeatures = tf.nn.dropout(globalFeatures, rate=1 - keep_prob_2)
    print("The global feature is {}".format(globalFeatures))
    globalFeatureN = (para.gcn_1_filter_n + para.gcn_2_filter_n) * 2

    # fully connected layer 1
    fc_layer_1 = fullyConnected(globalFeatures, inputFeatureN=globalFeatureN, outputFeatureN=para.fc_1_n)
    fc_layer_1 = tf.nn.relu(fc_layer_1)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, rate=1 - keep_prob_2)
    print("The output of the first fc layer is {}".format(fc_layer_1))

    # fully connected layer 2
    fc_layer_2 = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=para.outputClassN)
    print("The output of the second fc layer is {}".format(fc_layer_2))

    # =================================Define loss===========================
    predictSoftMax = tf.nn.softmax(fc_layer_2)
    predictLabels = tf.argmax(input=predictSoftMax, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer_2, labels=tf.stop_gradient(outputLabel))
    loss = tf.multiply(loss, weights)
    loss = tf.reduce_mean(input_tensor=loss)

    vars = tf.compat.v1.trainable_variables()
    loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 8e-6  # best: 8 #last: 10
    loss_total = loss + loss_reg

    correct_prediction = tf.equal(predictLabels, tf.argmax(input=outputLabel, axis=1))
    acc = tf.cast(correct_prediction, tf.float32)
    acc = tf.reduce_mean(input_tensor=acc)

    train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss_total)

    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim
        total_parameters += variable_parametes
    print('Total parameters number is {}'.format(total_parameters))

    trainOperation = {'train': train, 'loss_total': loss_total, 'loss': loss, 'acc': acc, 'loss_reg': loss_reg,
                     'inputPC': inputPC,
                     'inputGraph': inputGraph, 'outputLabel': outputLabel, 'weights': weights,
                     'predictLabels': predictLabels,
                     'keep_prob_1': keep_prob_1, 'keep_prob_2': keep_prob_2, 'lr': lr}

    return trainOperation


def trainOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion, weight_dict, learningRate):

    graphTrain_1 = inputGraph.tocsr()
    labelBinarize = label_binarize(inputLabel, classes=[j for j in range(para.outputClassN)])
    # shuffle arrays in a consistent way
    xTrain, graphTrain, labelTrain = shuffle(inputCoor, graphTrain_1, labelBinarize)

    batch_loss = []
    batch_acc = []
    batch_reg = []
    batchSize = para.batchSize
    for batchID in range(len(labelBinarize) // para.batchSize):
        start = batchID * batchSize
        end = start + batchSize
        batchCoor, batchGraph, batchLabel = get_mini_batch(xTrain, graphTrain, labelTrain, start, end)
        batchGraph = batchGraph.todense()
        batchCoor = add_noise(batchCoor, sigma=0.008, clip=0.02)
        batchWeight = uniform_weight(batchLabel)
        if para.weighting_scheme == 'weighted':
            batchWeight = weights_calculation(batchLabel, weight_dict)

        feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                     trainOperaion['outputLabel']: batchLabel, trainOperaion['lr']: learningRate,
                     trainOperaion['weights']: batchWeight,
                     trainOperaion['keep_prob_1']: para.keep_prob_1, trainOperaion['keep_prob_2']: para.keep_prob_2}

        opt, loss_train, acc_train, loss_reg_train = sess.run(
            [trainOperaion['train'], trainOperaion['loss_total'], trainOperaion['acc'], trainOperaion['loss_reg']],
            feed_dict=feed_dict)

        batch_loss.append(loss_train)
        batch_acc.append(acc_train)
        batch_reg.append(loss_reg_train)

    train_average_loss = np.mean(batch_loss)
    train_average_acc = np.mean(batch_acc)
    loss_reg_average = np.mean(batch_reg)

    return train_average_loss, train_average_acc, loss_reg_average


def evaluateOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion):

    xTest, graphTest, labelTest = inputCoor, inputGraph, inputLabel
    graphTest = graphTest.tocsr()
    labelBinarize = label_binarize(labelTest, classes=[i for i in range(para.outputClassN)])
    test_batch_size = para.testBatchSize

    test_loss = []
    test_acc = []
    test_predict = []

    for testBatchID in range(len(labelTest) // test_batch_size):
        start = testBatchID * test_batch_size
        end = start + test_batch_size
        batchCoor, batchGraph, batchLabel = get_mini_batch(xTest, graphTest, labelBinarize, start, end)
        batchWeight = uniform_weight(batchLabel)
        batchGraph = batchGraph.todense()

        feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                     trainOperaion['outputLabel']: batchLabel, trainOperaion['weights']: batchWeight,
                     trainOperaion['keep_prob_1']: 1.0, trainOperaion['keep_prob_2']: 1.0}

        predict, loss_test, acc_test = sess.run(
            [trainOperaion['predictLabels'], trainOperaion['loss'], trainOperaion['acc']], feed_dict=feed_dict)
        test_loss.append(loss_test)
        test_acc.append(acc_test)
        test_predict.append(predict)

    test_average_loss = np.mean(test_loss)
    test_average_acc = np.mean(test_acc)

    return test_average_loss, test_average_acc, test_predict