"""
This is a class to compare the prediction result of different experiments
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# load prediction result
resultDir = 'result_folder'
dataDir = 'test'
classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}

testdata = []
for cls in os.listdir(dataDir):
    for data in os.listdir(os.path.join(dataDir, cls)):
        testdata.append(os.path.join(dataDir, cls, data))
tdataDir = 'train'
traindata = []
for cls in os.listdir(tdataDir):
    for data in os.listdir(os.path.join(tdataDir, cls)):
        traindata.append(os.path.join(tdataDir, cls, data))


def read_result(file):
    result = pd.read_csv(file, sep='\t', header=0)
    return result


def compare(file1, file2, outname):
    df1 = read_result(os.path.join(resultDir, file1))
    df2 = read_result(os.path.join(resultDir, file2))
    if df1.shape != df2.shape:
        print('compare pair are with different shape')
    else:
        df_com = df1
        df_com['pred2'] = df2['pred']
        df_com['different'] = np.where(df1['pred'] != df2['pred'], True, False)
        df_com['file'] = testdata
        df_com[df_com.different].to_csv(os.path.join(resultDir, f'{outname}.txt'))
        return df_com[df_com.different]


def output_wrong(filename):
    df = read_result(os.path.join(resultDir, filename))
    df['wrong'] = np.where(df['pred'] != df['real'], True, False)
    df['file'] = testdata[:df.shape[0]]
    df[df.wrong].to_csv(os.path.join(resultDir, f"{filename.replace('.txt', '')}_wrong.txt"))
    return df[df.wrong]


def main():
    fout = open(os.path.join(resultDir, 'testdatafile_relabel0304.txt'), 'w')
    fout.write(f'no\tfilepath\n')
    for i, test in enumerate(testdata):
        fout.write(f'{i}\t{test}\n')


if __name__ == '__main__':
    main()

