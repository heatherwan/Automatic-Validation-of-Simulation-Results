#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:53:31 2017

@author: yingxuezhang
"""
import h5py
import numpy as np
import scipy


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def loadDataFile(filename):
    return load_h5(filename)


def load_h5_sf(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    sf = f['sf'][:]
    return data, sf, label


def loadDataFile_sf(filename):
    return load_h5_sf(filename)


def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape  # M = #points K = #neighbor
    assert M, k == idx.shape
    assert dist.min() >= 0
    # Weights.
    sigma2 = np.mean(dist[:, -1]) ** 2
    # print sigma2
    dist = np.exp(- dist ** 2 / sigma2)  # normalized distance

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))  # V is the data to fill I,J is the indices to put inside
    # No self-connections.
    W.setdiag(0)

    # undirected graph. use the direction with bigger weight
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W


def normalize_adj(adj):
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalized_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    norm_laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    return norm_laplacian


def scaled_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian


def get_mini_batch(x_signal, graph, y, start, end):
    return x_signal[start:end], graph[start:end], y[start:end]


def add_noise(batch_data, sigma=0.015, clip=0.05):
    batch_n, nodes_n_1, feature_n = batch_data.shape
    noise = np.clip(sigma * np.random.randn(batch_n, nodes_n_1, feature_n), -1 * clip, clip)
    new_data = batch_data + noise
    return new_data


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
    batch_labels = np.argmax(batch_labels, axis=1)

    for i in batch_labels:
        weights.append(weight_dict[i])
    return weights


def uniform_weight(trainLabel):
    weights = []
    [weights.append(1) for i in range(len(trainLabel))]
    return weights
