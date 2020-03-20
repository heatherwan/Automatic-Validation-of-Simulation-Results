import os
import shutil


class Parameters:
    def __init__(self):
        self.gpu = '-1'
        self.model = 'pointnet_cls'
        self.log_dir = 'log'
        self.logmodelDir = os.path.join(os.getcwd(), 'logmodel')
        self.logDir = os.path.join(os.getcwd(), 'log')
        self.dataDir = os.path.join(os.getcwd(), 'datasets')

        if not os.path.isdir(self.logDir):
            os.mkdir(self.logDir)
        else:
            shutil.move(os.path.join(self.logDir, 'train'), os.path.join(self.logDir, 'trainold'))
            shutil.move(os.path.join(self.logDir, 'test'), os.path.join(self.logDir, 'testold'))

        if not os.path.isdir(self.logmodelDir):
            os.mkdir(self.logmodelDir)
        self.TRAIN_FILES = os.path.join(self.dataDir, 'traindataset_dim4_480.hdf5')
        self.TEST_FILES = os.path.join(self.dataDir, 'testdataset_dim4_160.hdf5')

        self.outputClassN = 4
        self.pointNumber = 1024
        self.dim = 4

        self.batchSize = 32
        self.testBatchSize = 32
        self.max_epoch = 200
        self.learningRate = 2e-3

        self.weighting_scheme = 'weighted'
        self.weight_scaler = 4  # 50
        self.expName = f'exp203_point{self.pointNumber}_batch{self.batchSize}_out{self.outputClassN}_weighted{self.weight_scaler}'  # save model path

        self.momentum = 0.9
        self.optimizer = 'adam'  # or momentum
        self.decay_step = 20000  # 1 epoch 1000 step
        self.decay_rate = 0.7
