import os


class Parameters:
    def __init__(self):
        self.gpu = False

        self.modelDir = os.path.join(os.getcwd(), 'model')
        self.logDir = os.path.join(os.getcwd(), 'log')
        self.dataDir = os.path.join(os.getcwd(), 'data')
        self.graphDir = os.path.join(os.getcwd(), 'graph')
        if not os.path.isdir(self.logDir):
            os.mkdir(self.logDir)
        if not os.path.isdir(self.modelDir):
            os.mkdir(self.modelDir)
        if not os.path.isdir(self.graphDir):
            os.mkdir(self.graphDir)

        self.trainDataset = 'traindataset_dim4_480_3cat.hdf5'
        self.testDataset = 'testdataset_dim4_160_3cat.hdf5'
        self.samplingType = 'farthest_sampling'
        self.neighborNumber = 4
        self.outputClassN = 3
        self.pointNumber = 1024
        self.dim = 4
        self.gcn_1_filter_n = 1000
        self.gcn_2_filter_n = 1000
        self.gcn_3_filter_n = 1000
        self.fc_1_n = 600
        self.chebyshev_1_Order = 4
        self.chebyshev_2_Order = 3
        self.keep_prob_1 = 0.9  # 0.9 original
        self.keep_prob_2 = 0.55
        self.batchSize = 32
        self.testBatchSize = 1
        self.max_epoch = 100
        self.learningRate = 1e-3
        self.weighting_scheme = 'weighted'
        self.weight_scaler = 4  # 50
        self.expCount = '001'
        self.expName = f'exp{self.expCount}_point{self.pointNumber}_nn{self.neighborNumber}_cheby_{self.chebyshev_1_Order}' \
                       f'_{self.chebyshev_2_Order}_out{self.outputClassN}'  # save model path
