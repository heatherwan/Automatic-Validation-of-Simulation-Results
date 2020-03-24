#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:25:59 2017

@author: yingxuezhang
"""
import os
import utils as utils
import scipy
from utils import adjacency, scaled_laplacian
import numpy as np
from scipy.spatial import cKDTree
import pickle
from Parameters import Parameters

para = Parameters(evaluation=False)


def farthestSampling(file_name, NUM_POINT):

    # current_data, current_label = utils.loadDataFile(file_name)
    current_data, current_sf, current_label = utils.loadDataFile_sf(file_name)
    current_data = current_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(current_label)
    current_label = np.int_(current_label)

    return current_data, current_sf, current_label


def uniformSampling(file_names, NUM_POINT):
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    for index in range(len(file_indexs)):
        current_data, current_label = utils.loadDataFile(file_names[file_indexs[index]])
        current_label = np.squeeze(current_label)
        current_label = np.int_(current_label)
        output = np.zeros((len(current_data), NUM_POINT, 3))
        for i, object_xyz in enumerate(current_data):
            samples_index = np.random.choice(2048, NUM_POINT, replace=False)
            output[i] = object_xyz[samples_index]
        inputData.update({index: output})
        inputLabel.update({index: current_label})
    return inputData, inputLabel


# ModelNet40 official train/test split
def load_data(NUM_POINT, sampleType):
    # define datafiles to read
    TRAIN_FILE = os.path.join(para.dataDir, para.trainDataset)
    TEST_FILE = os.path.join(para.dataDir, para.testDataset)

    if sampleType == 'farthest_sampling':
        inputTrainFarthest, inputTrainSF, inputTrainLabel = farthestSampling(TRAIN_FILE, NUM_POINT)
        inputTestFathest, inputTestSF, inputTestLabel = farthestSampling(TEST_FILE, NUM_POINT)
        return inputTrainFarthest, inputTrainSF, inputTrainLabel, inputTestFathest, inputTestSF, inputTestLabel

    elif sampleType == 'uniform_sampling':
        inputTrainFarthest, inputTrainLabel = uniformSampling(TRAIN_FILE, NUM_POINT)
        inputTestFathest, inputTestLabel = uniformSampling(TEST_FILE, NUM_POINT)

        return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel


# generate graph structure and store in the system
def prepareGraph(inputData, neighborNumber, pointNumber, dataType):

    global batchFlattenLaplacian
    save_graph_path = os.path.join(para.graphDir, para.expName + dataType)

    if not os.path.isfile(save_graph_path):
        print("calculating the graph data")

        for i in range(len(inputData)):
            print(f'generate graph for object: {i} in {dataType} dataset')
            pcCoordinates = inputData[i]
            tree = cKDTree(pcCoordinates)
            dd, ii = tree.query(pcCoordinates, k=neighborNumber)
            A = adjacency(dd, ii)
            scaledLaplacian = scaled_laplacian(A)
            flattenLaplacian = scaledLaplacian.tolil().reshape((1, pointNumber * pointNumber))
            if i == 0:
                batchFlattenLaplacian = flattenLaplacian
            else:
                batchFlattenLaplacian = scipy.sparse.vstack([batchFlattenLaplacian, flattenLaplacian])

        with open(save_graph_path, 'wb') as handle:
            pickle.dump(batchFlattenLaplacian, handle)
        return batchFlattenLaplacian
    else:
        print(f"Loading the graph {dataType} data from {para.expName}")
        scaledLaplacian = loadGraph(save_graph_path)
        return scaledLaplacian


def loadGraph(filePath):
    with open(filePath, 'rb') as handle:
        graph = pickle.load(handle)
        return graph


def prepareData(inputTrain, inputTest, neighborNumber, pointNumber):
    scaledLaplacianTrain = prepareGraph(inputTrain, neighborNumber, pointNumber, 'train', )
    scaledLaplacianTest = prepareGraph(inputTest, neighborNumber, pointNumber, 'test')
    return scaledLaplacianTrain, scaledLaplacianTest
