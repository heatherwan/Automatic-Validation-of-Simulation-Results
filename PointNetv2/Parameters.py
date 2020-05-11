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
        self.model = 'ldgcnn'  # 'pointnet_cls'# 'dgcnn' # 'lgdcnn'
        self.outputClassN = 4
        self.pointNumber = 1024
        self.dim = 5  # 3 coordinate, 1 safety factor, 1 distance from Min SF, 3 normals
        self.batchSize = 2
        self.testBatchSize = 2
        self.max_epoch = 200
        self.learningRate = 2e-3
        self.momentum = 0.9
        self.optimizer = 'adam'  # or momentum
        self.decay_step = 20000  # 1 epoch 1000 step
        self.decay_rate = 0.7
        self.weight_scaler = 0  # 0 = no weight

        # LDGCNN The parameters of retrained classifier
        self.class_model = 'ldgcnn_classifier'
        self.class_feature = 1024
        self.class_max_epoch = 100
        self.class_optimizer = 'momentum'

        self.expName = f'exp423_point{self.pointNumber}_batch{self.batchSize}_out{self.outputClassN}' \
                       f'_weighted{self.weight_scaler}'  # save model path

        # ==============Files setting===========================
        self.logmodelDir = 'logmodel'
        if not os.path.isdir(self.logmodelDir):
            os.mkdir(self.logmodelDir)
        if not evaluation:
            self.logDir = 'log'
            if not os.path.isdir(self.logDir):
                os.mkdir(self.logDir)
                # os.mkdir(os.path.join(self.logDir, f'{self.expName[:6]}train'))
                # os.mkdir(os.path.join(self.logDir, f'{self.expName[:6]}test'))
                # os.mkdir(os.path.join(self.logDir, f'{self.expName[:6]}trainold'))
                # os.mkdir(os.path.join(self.logDir, f'{self.expName[:6]}testold'))
            # else:
            #     for file in os.listdir(os.path.join(self.logDir, f'{self.expName[:6]}train')):
            #         shutil.move(os.path.join(self.logDir, f'{self.expName[:6]}train', file),
            #                     os.path.join(self.logDir, f'{self.expName[:6]}trainold', file))
            #     for file in os.listdir(os.path.join(self.logDir, f'{self.expName[:6]}test')):
            #         shutil.move(os.path.join(self.logDir, f'{self.expName[:6]}test', file),
            #                     os.path.join(self.logDir, f'{self.expName[:6]}testold', file))
        else:
            self.evallog = 'evallog'
            if not os.path.isdir(self.evallog):
                os.mkdir(self.evallog)

        self.dataDir = os.path.join(BASE_DIR, 'datasets')
        self.TRAIN_FILES = [os.path.join(self.dataDir, 'traindataset_651_1024_dim5.hdf5')]
        self.TEST_FILES = [os.path.join(self.dataDir, 'testdataset_163_1024_dim5.hdf5')]

        self.classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}
        # self.classes = {1: 'EM13_contactradius', 2: 'EM4_hole', 0: 'Good'}

