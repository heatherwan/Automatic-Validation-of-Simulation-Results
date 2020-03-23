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
        self.model = 'pointnet_cls'
        self.outputClassN = 4
        self.pointNumber = 1024
        self.dim = 4
        self.batchSize = 32
        self.testBatchSize = 32
        self.max_epoch = 2
        self.learningRate = 2e-3
        self.momentum = 0.9
        self.optimizer = 'adam'  # or momentum
        self.decay_step = 20000  # 1 epoch 1000 step
        self.decay_rate = 0.7
        self.weighting_scheme = 'weighted'
        self.weight_scaler = 4  # 50

        # ==============Files setting===========================
        # self.logmodelDir = os.path.join(os.getcwd(), 'logmodel')
        # if not os.path.isdir(self.logmodelDir):
        #     os.mkdir(self.logmodelDir)
        # if not evaluation:
        #     self.logDir = os.path.join(os.getcwd(), 'log')
        #     if not os.path.isdir(self.logDir):
        #         os.mkdir(self.logDir)
        #         os.mkdir(os.path.join(self.logDir, 'train'))
        #         os.mkdir(os.path.join(self.logDir, 'test'))
        #         os.mkdir(os.path.join(self.logDir, 'trainold'))
        #         os.mkdir(os.path.join(self.logDir, 'testold'))
        #     else:
        #         for file in os.listdir(os.path.join(self.logDir, 'train')):
        #             shutil.move(os.path.join(self.logDir, 'train', file), os.path.join(self.logDir, 'trainold', file))
        #         for file in os.listdir(os.path.join(self.logDir, 'test')):
        #             shutil.move(os.path.join(self.logDir, 'test', file), os.path.join(self.logDir, 'testold', file))
        # else:
        #     self.evallog = os.path.join(os.getcwd(), 'evallog')
        #     if not os.path.isdir(self.evallog):
        #         os.mkdir(self.evallog)
        #
        # self.dataDir = os.path.join(os.getcwd(), 'datasets')
        # self.TRAIN_FILES = os.path.join(self.dataDir, 'traindataset_dim4_480.hdf5')
        # self.TEST_FILES = os.path.join(self.dataDir, 'testdataset_dim4_160.hdf5')

        self.logmodelDir = 'logmodel'
        if not os.path.isdir(self.logmodelDir):
            os.mkdir(self.logmodelDir)
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

        self.dataDir = os.path.join(BASE_DIR, 'datasets')
        self.TRAIN_FILES = os.path.join(self.dataDir, 'traindataset_dim4_480.hdf5')
        self.TEST_FILES = os.path.join(self.dataDir, 'testdataset_dim4_160.hdf5')

        self.expName = f'exp203_point{self.pointNumber}_batch{self.batchSize}_out{self.outputClassN}' \
                       f'_weighted{self.weight_scaler}'  # save model path

        self.classes = {1: 'EM1', 2: 'EM3', 3: 'EM4', 0: 'Good'}
