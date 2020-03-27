import tensorflow as tf
import numpy as np

import utils
import layers
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize


# ===========================Hyper parameter=====================
def model_architecture(para):
    inputPC = tf.compat.v1.placeholder(tf.float32, [None, para.pointNumber, para.dim])
    inputGraph = tf.compat.v1.placeholder(tf.float32, [None, para.pointNumber * para.pointNumber])
    outputLabel = tf.compat.v1.placeholder(tf.float32, [None, para.outputClassN])

    # Note the global_step=batch parameter to minimize.
    # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
    batch = tf.Variable(0)
    scaledLaplacian = tf.reshape(inputGraph, [-1, para.pointNumber, para.pointNumber])

    weights = tf.compat.v1.placeholder(tf.float32, [None])
    lr = tf.compat.v1.placeholder(tf.float32)
    dropout_prob_1 = tf.compat.v1.placeholder(tf.float32)
    dropout_prob_2 = tf.compat.v1.placeholder(tf.float32)

    # gcn layer 1
    gcn_1 = layers.gcnLayer(inputPC, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.dim,
                            outputFeatureN=para.gcn_1_filter_n,
                            chebyshev_order=para.chebyshev_1_Order)
    gcn_1_output = tf.nn.dropout(gcn_1, rate=dropout_prob_1)
    gcn_1_pooling = layers.globalPooling(gcn_1_output)

    # gcn_layer_2

    gcn_2 = layers.gcnLayer(gcn_1_output, scaledLaplacian, pointNumber=para.pointNumber,
                            inputFeatureN=para.gcn_1_filter_n,
                            outputFeatureN=para.gcn_2_filter_n,
                            chebyshev_order=para.chebyshev_2_Order)
    gcn_2_output = tf.nn.dropout(gcn_2, rate=dropout_prob_1)
    gcn_2_pooling = layers.globalPooling(gcn_2_output)

    # concatenate global features
    globalFeatures = tf.concat([gcn_1_pooling, gcn_2_pooling], axis=1)
    globalFeatures = tf.nn.dropout(globalFeatures, rate=dropout_prob_2)
    globalFeatureNum = (para.gcn_1_filter_n + para.gcn_2_filter_n) * 2  # 1 max pooling, 1 variance pooling

    # fully connected layer 1
    fc_layer_1 = layers.fullyConnected(globalFeatures, inputFeatureN=globalFeatureNum, outputFeatureN=para.fc_1_n)
    fc_layer_1 = tf.nn.relu(fc_layer_1)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, rate=dropout_prob_2)

    # fully connected layer 2
    fc_layer_2 = layers.fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=para.outputClassN)

    # =================================Define loss===========================
    predictSoftMax = tf.nn.softmax(fc_layer_2)
    predictLabels = tf.argmax(input=predictSoftMax, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer_2, labels=tf.stop_gradient(outputLabel))
    loss = tf.multiply(loss, weights)
    loss = tf.reduce_mean(input_tensor=loss)  # mean loss

    variables = tf.compat.v1.trainable_variables()
    loss_reg = tf.add_n(
        [tf.nn.l2_loss(v) for v in variables if 'bias' and 'Variable:0' not in v.name]) * 8e-6
    loss_total = loss + loss_reg
    tf.compat.v1.summary.scalar('mean classify loss', loss_total)

    correct_prediction = tf.equal(predictLabels, tf.argmax(input=outputLabel, axis=1))
    acc = tf.cast(correct_prediction, tf.float32)
    acc = tf.reduce_mean(input_tensor=acc)
    tf.compat.v1.summary.scalar('accuracy', acc)

    train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss_total, global_step=batch)

    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim
        total_parameters += variable_parametes
    print(f'Total parameters number is {total_parameters}')

    merged = tf.compat.v1.summary.merge_all()
    trainOperation = {'train': train, 'loss_total': loss_total, 'loss': loss, 'acc': acc, 'loss_reg': loss_reg,
                      'inputPC': inputPC,
                      'inputGraph': inputGraph, 'outputLabel': outputLabel, 'weights': weights,
                      'predictLabels': predictLabels,
                      'dropout_prob_1': dropout_prob_1, 'dropout_prob_2': dropout_prob_2, 'lr': lr,
                      'merged': merged, 'step': batch}

    return trainOperation


def trainOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperation, weight_dict, learningRate,
                  train_writer):
    graphTrain_1 = inputGraph.tocsr()
    labelBinarize = label_binarize(inputLabel, classes=[j for j in range(para.outputClassN)])
    # shuffle arrays in a consistent way
    xTrain, graphTrain, labelTrain = shuffle(inputCoor, graphTrain_1, labelBinarize)

    batch_loss = []
    batch_acc = []
    batchSize = para.batchSize
    for batchID in range(len(labelBinarize) // para.batchSize):
        start = batchID * batchSize
        end = start + batchSize
        batchCoor, batchGraph, batchLabel = utils.get_mini_batch(xTrain, graphTrain, labelTrain, start, end)
        batchGraph = batchGraph.todense()
        batchCoor = utils.add_noise(batchCoor, sigma=0.008, clip=0.02)

        batchWeight = utils.weights_calculation(batchLabel, weight_dict)

        feed_dict = {trainOperation['inputPC']: batchCoor, trainOperation['inputGraph']: batchGraph,
                     trainOperation['outputLabel']: batchLabel, trainOperation['lr']: learningRate,
                     trainOperation['weights']: batchWeight,
                     trainOperation['dropout_prob_1']: para.dropout_prob_1,
                     trainOperation['dropout_prob_2']: para.dropout_prob_2}

        summary, step, _, loss_train, acc_train = sess.run(
            [trainOperation['merged'], trainOperation['step'], trainOperation['train'],
             trainOperation['loss_total'], trainOperation['acc']],
            feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        batch_loss.append(loss_train)
        batch_acc.append(acc_train)

    train_average_loss = np.mean(batch_loss)  # avg. total loss
    train_average_acc = np.mean(batch_acc)

    return train_average_loss, train_average_acc


def evaluateOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperation, test_writer):
    xTest, graphTest, labelTest = inputCoor, inputGraph, inputLabel
    graphTest = graphTest.tocsr()
    labelBinarize = label_binarize(labelTest, classes=[i for i in range(para.outputClassN)])
    test_batch_size = para.batchSize

    test_loss = []
    test_acc = []
    test_predict = []

    for testBatchID in range(len(labelTest) // test_batch_size):
        start = testBatchID * test_batch_size
        end = start + test_batch_size
        batchCoor, batchGraph, batchLabel = utils.get_mini_batch(xTest, graphTest, labelBinarize, start, end)
        batchWeight = utils.uniform_weight(batchLabel)
        batchGraph = batchGraph.todense()

        feed_dict = {trainOperation['inputPC']: batchCoor, trainOperation['inputGraph']: batchGraph,
                     trainOperation['outputLabel']: batchLabel, trainOperation['weights']: batchWeight,
                     trainOperation['dropout_prob_1']: 1.0, trainOperation['dropout_prob_2']: 1.0}

        summary, step, predict, loss_test, acc_test = sess.run(
            [trainOperation['merged'], trainOperation['step'], trainOperation['predictLabels'],
             trainOperation['loss_total'], trainOperation['acc']],
            feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        test_loss.append(loss_test)
        test_acc.append(acc_test)
        test_predict.append(predict)

    test_average_loss = np.mean(test_loss)  # avg. total loss
    test_average_acc = np.mean(test_acc)

    return test_average_loss, test_average_acc, test_predict
