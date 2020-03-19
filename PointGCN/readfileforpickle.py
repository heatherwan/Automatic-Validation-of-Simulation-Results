import pickle
import numpy as np
from sklearn.preprocessing import label_binarize
from collections import Counter

filename = ['log/exp01_point1024_nn4_cheby_4_3_dim4_confusion_mat',
            'log/exp01_point1024_nn4_cheby_4_3_dim4_mean_class_acc_record']
content = []
for file in filename:
    with open(file, 'rb') as pickle_file:
        content.append(pickle.load(pickle_file))
for i in range(len(content[0])):
    print(f'epoch: {i}')
    print(content[0][i])
    print(content[1][i])

