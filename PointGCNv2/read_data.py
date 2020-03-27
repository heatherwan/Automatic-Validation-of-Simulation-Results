#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:39:59 2020

@author: wantinglin
"""
import os
import pickle
import sys

import numpy as np
import scipy
from scipy.spatial import cKDTree

import utils
from Parameters import Parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
para = Parameters(evaluation=False)


def load_data(isEvaluation=False):
    # define datafiles to read
    TRAIN_FILE = os.path.join(para.dataDir, para.trainDataset)
    TEST_FILE = os.path.join(para.dataDir, para.testDataset)

    if isEvaluation:
        inputTest, inputTestOther, current_label = utils.loadDataFile_other(TEST_FILE)
        current_label = np.squeeze(current_label)
        inputTestLabel = np.int_(current_label)
        return inputTest, inputTestOther, inputTestLabel
    else:
        inputTrain, inputTrainOther, current_label = utils.loadDataFile_other(TRAIN_FILE)
        current_label = np.squeeze(current_label)
        inputTrainLabel = np.int_(current_label)

        inputTest, inputTestOther, current_label = utils.loadDataFile_other(TEST_FILE)
        current_label = np.squeeze(current_label)
        inputTestLabel = np.int_(current_label)

        return inputTrain, inputTrainOther, inputTrainLabel, inputTest, inputTestOther, inputTestLabel


def loadGraph(filePath):
    with open(filePath, 'rb') as handle:
        graph = pickle.load(handle)
        return graph


# generate graph structure and store in the system
def prepareGraph(inputData, neighborNumber, pointNumber, train_test):

    global batchFlattenLaplacian
    save_graph_path = os.path.join(para.graphDir, para.expName[7:] + train_test)
    if not os.path.isfile(save_graph_path):
        print("calculating the graph data")

        for i in range(len(inputData)):
            print(f'generate graph for object: {i} in {train_test} dataset')
            pcCoordinates = inputData[i]
            tree = cKDTree(pcCoordinates)
            dd, ii = tree.query(pcCoordinates, k=neighborNumber)
            A = utils.adjacency(dd, ii)
            scaledLaplacian = utils.scaled_laplacian(A)
            flattenLaplacian = scaledLaplacian.tolil().reshape((1, pointNumber * pointNumber))
            if i == 0:
                batchFlattenLaplacian = flattenLaplacian
            else:
                batchFlattenLaplacian = scipy.sparse.vstack([batchFlattenLaplacian, flattenLaplacian])

        with open(save_graph_path, 'wb') as handle:
            pickle.dump(batchFlattenLaplacian, handle)
        return batchFlattenLaplacian
    else:
        print(f"Loading the graph {train_test} data from {para.expName}")
        scaledLaplacian = loadGraph(save_graph_path)
        return scaledLaplacian


def prepareData(inputTrain, inputTest, neighborNumber, pointNumber, isEvaluation=False):
    if isEvaluation:
        scaledLaplacianTest = prepareGraph(inputTest, neighborNumber, pointNumber, 'test')
        return scaledLaplacianTest
    else:
        scaledLaplacianTrain = prepareGraph(inputTrain, neighborNumber, pointNumber, 'train', )
        scaledLaplacianTest = prepareGraph(inputTest, neighborNumber, pointNumber, 'test')
        return scaledLaplacianTrain, scaledLaplacianTest
