import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


class Parameters:
    def __init__(self, evaluation=False):
        # ==============Network setting===========================
        self.gpu = False
        self.model = 'dgcnn'  # 'pointnet_cls'# 'dgcnn' # 'lgdcnn' # 'lgdcnn_cls' #'densepoint_tf'
        self.outputClassN = 4
        self.pointNumber = 1024
        self.dim = 6  # 3 coordinate, 1 safety factor, 1 distance from Min SF, 1 minSF, 3 normals, 2 curcatures
        self.batchSize = 25
        self.testBatchSize = 25
        self.max_epoch = 200
        self.learningRate = 2e-3
        self.momentum = 0.9
        self.optimizer = 'adam'  # or momentum
        self.decay_step = 20000  # 20000/#sample = around 30 epoch update once
        self.decay_rate = 0.7

        # parameters for loss setting
        self.binary_loss = True
        self.loss_smoothing = 0.05
        self.fine_coarse_mapping = [[1.0, 0.0],
                                    [0.0, 1.0],
                                    [0.0, 1.0],
                                    [0.0, 1.0]]

        # parameters for cross validation
        self.validation = False
        self.split_num = 5

        # parameter for dgcnn and ldgcnn
        self.k = 20

        # parameters for densepoint
        self.group_num = 4
        self.k_add = 24

        self.expName = f'exp433_point{self.pointNumber}_batch{self.batchSize}_out{self.outputClassN}' \
                       f'_smooth{self.loss_smoothing}'  # save model path

        # ==============Files setting===========================
        self.logmodelDir = 'logmodel'
        if not os.path.isdir(self.logmodelDir):
            os.mkdir(self.logmodelDir)
        if not evaluation:
            self.logDir = 'log'
            if not os.path.isdir(self.logDir):
                os.mkdir(self.logDir)
        else:
            self.evallog = 'evallog'
            if not os.path.isdir(self.evallog):
                os.mkdir(self.evallog)

        self.dataDir = os.path.join(BASE_DIR, 'datasets')
        self.TRAIN_FILES = [os.path.join(self.dataDir, 'traindataset_651_1024_dim6_normal.hdf5')]
        self.TEST_FILES = [os.path.join(self.dataDir, 'testdataset_163_1024_dim6_normal.hdf5')]

        self.classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}
        # self.classes = {1: 'EM13_contactradius', 2: 'EM4_hole', 0: 'Good'}
