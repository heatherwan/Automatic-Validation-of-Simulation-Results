import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
import pickle

def reliability(y, y_prob):
    nbin = 10
    # class0
    y_true = y.copy()
    y_true[y_true == 0] = 4
    y_true[y_true != 4] = 0
    y_true[y_true == 4] = 1
    select = y_prob[:, 0]
    print(select)
    x0, y0 = calibration_curve(y_true, select, n_bins=nbin)
    # class 1
    y_true = y.copy()
    y_true[y_true != 1] = 0
    select = y_prob[:, 1]
    x1, y1 = calibration_curve(y_true, select, n_bins=nbin)
    # class 2
    y_true = y.copy()
    y_true[y_true != 2] = 0
    select = y_prob[:, 2]
    x2, y2 = calibration_curve(y_true, select, n_bins=nbin)
    # class 3
    y_true = y.copy()
    y_true[y_true != 3] = 0
    select = y_prob[:, 3]
    x3, y3 = calibration_curve(y_true, select, n_bins=nbin)
    x = np.linspace(0, 1, 101)

    fig = plt.figure()
    plt.plot(x, x, label='Identity', color='black')
    plt.plot(y0, x0, label='Good')
    plt.plot(y1, x1, label='EM1_Contact')
    plt.plot(y2, x2, label='EM3_Radius')
    plt.plot(y3, x3, label='EM4_Hole')
    plt.legend()
    plt.show()


def main():

    # file = 'Input_logit/exp143_test_logit.txt'
    # data = pd.read_csv(file, sep='\t', header=0, index_col=False)
    # y_pred = np.array(data.iloc[:, 3:])
    # y_prob = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1).reshape(149, 1)
    # y_true = np.array(data.iloc[:, 2])
    # reliability(y_true, y_prob)
    #
    # # import calibrated result
    # # file = 'result/exp145_TS_test.txt'
    # # data = pd.read_csv(file, header=None, sep=' ')
    # # y_prob = np.array(data.iloc[149:])
    # # print(y_prob)
    # # reliability(y_true, y_prob)
    #
    # # import calibrated result
    # file = 'result/exp145_DIR-ODIR_test_0.0025_0.01.txt'
    # data = pd.read_csv(file, header=None, sep=' ')
    # y_prob = np.array(data.iloc[149:])
    # print(y_prob)
    # reliability(y_true, y_prob)
    #
    # # import calibrated result
    # file = 'result/exp145_MS-ODIR_test_0.0025_0.01.txt'
    # data = pd.read_csv(file, header=None, sep=' ')
    # y_prob = np.array(data.iloc[149:])
    # print(y_prob)
    # reliability(y_true, y_prob)

    # get weight
    filename = 'model_weights/model_MS-ODIR_exp144_l2=0.0025_mu=0.01.p'
    file = open(filename, 'rb')
    model = pickle.load(file)
    W = model[0][-1][0]
    b = model[0][-1][1]
    print(W, b)


if __name__ == '__main__':
    main()
