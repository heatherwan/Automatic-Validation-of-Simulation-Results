#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:39:59 2020

@author: wantinglin
"""
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class Parameters:
    def __init__(self, evaluation):
        # ==============Network setting===========================
        self.classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}
        # self.classes = {1: 'EM13_contactradius', 2: 'EM4_hole', 0: 'Good'}
        self.gpu = False
        self.samplingType = 'farthest_sampling'
        self.neighborNumber = 4
        self.outputClassN = 4
        self.pointNumber = 1024
        self.dim = 5
        self.gcn_1_filter_n = 1000
        self.gcn_2_filter_n = 1000
        self.fc_1_n = 600
        self.chebyshev_1_Order = 4
        self.chebyshev_2_Order = 3
        self.dropout_prob_1 = 0.1  # 0.9 original
        self.dropout_prob_2 = 0.45
        self.batchSize = 32
        self.testBatchSize = 1
        self.max_epoch = 10
        self.learningRate = 1e-3
        self.weight_scaler = 4  # 0 = no weight

        # ==============Files setting===========================
        if not evaluation:
            self.logDir = 'log'
            if not os.path.isdir(self.logDir):
                os.mkdir(self.logDir)
                os.mkdir(os.path.join(self.logDir, 'train'))
                os.mkdir(os.path.join(self.logDir, 'test'))
                os.mkdir(os.path.join(self.logDir, 'trainold'))
                os.mkdir(os.path.join(self.logDir, 'testold'))
            else:
                for file in os.listdir(os.path.join(self.logDir, 'train')):
                    shutil.move(os.path.join(self.logDir, 'train', file), os.path.join(self.logDir, 'trainold', file))
                for file in os.listdir(os.path.join(self.logDir, 'test')):
                    shutil.move(os.path.join(self.logDir, 'test', file), os.path.join(self.logDir, 'testold', file))
        else:
            self.evallog = 'evallog'
            if not os.path.isdir(self.evallog):
                os.mkdir(self.evallog)
        self.dataDir = 'data'
        self.modelDir = 'model'
        self.logDir = 'log'
        self.graphDir = 'graph'
        if not os.path.isdir(self.logDir):
            os.mkdir(self.logDir)
        if not os.path.isdir(self.modelDir):
            os.mkdir(self.modelDir)
        if not os.path.isdir(self.graphDir):
            os.mkdir(self.graphDir)
        self.trainDataset = 'traindataset_dim4_480_1024_dist.hdf5'
        self.testDataset = 'testdataset_dim4_160_1024_dist.hdf5'
        self.expCount = '011'
        self.expName = f'exp{self.expCount}_point{self.pointNumber}_nn{self.neighborNumber}_cheby_{self.chebyshev_1_Order}' \
                       f'_{self.chebyshev_2_Order}_out{self.outputClassN}'  # save model path
