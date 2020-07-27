import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


class Parameters:
    def __init__(self, evaluation=False):
        # ==============Network setting===========================
        self.gpu = True
        self.model = 'ldgcnn'  # 'pointnet_cls'# 'dgcnn' # 'lgdcnn' # 'lgdcnn_cls' #'densepoint_tf'
        self.outputClassN = 5
        self.pointNumber = 2048
        self.dim = 6  # 3 coordinate, 1 safety factor, 1 distance from Min SF, 1 minSF, 3 normals, 2 curcatures
        self.batchSize = 16
        self.testBatchSize = 16
        self.max_epoch = 70
        self.learningRate = 2e-3
        self.momentum = 0.9
        self.optimizer = 'adam'  # or momentum
        self.decay_step = 20000  # 20000/#sample = around 30 epoch update once
        self.decay_rate = 0.7
        self.num_votes = 5
        self.continue_train = False

        # parameters for loss setting
        self.binary_loss = True
        self.loss_smoothing = 0.1
        if self.outputClassN == 5:
            self.fine_coarse_mapping = [[1.0, 0.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0]]
        elif self.outputClassN == 4:
            self.fine_coarse_mapping = [[1.0, 0.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0]]
        elif self.outputClassN ==3:
            self.fine_coarse_mapping = [[1.0, 0.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0]]

        # parameters for cross validation
        self.validation = False
        self.split_num = 5

        # parameter for dgcnn and ldgcnn
        self.k = 10

        # parameters for densepoint
        self.group_num = 4
        self.k_add = 24

        self.expName = f'exp610_point{self.pointNumber}_batch{self.batchSize}_out{self.outputClassN}' \
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
        self.TRAIN_FILES = os.path.join(self.dataDir, 'traindataset_SF_2048.hdf5')
        self.TEST_FILES = os.path.join(self.dataDir, 'validdataset_SF_2048.hdf5')

        if self.outputClassN == 5:
            self.classes = {1: 'EM1_contact', 2: 'EM2_inhole', 3: 'EM3_radius', 4: 'EM4_hole', 0: 'Good'}
        elif self.outputClassN == 4:
            self.classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}
        elif self.outputClassN == 3:
            self.classes = {1: 'EM1_contact', 2: 'EM4_hole', 0: 'Good'}
