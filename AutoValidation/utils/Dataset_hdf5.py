"""
    Model to read a list of .hdf5 files

"""

import os
import sys
import numpy as np
from utils import provider
from imblearn.over_sampling import RandomOverSampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
DATADIR = os.path.join(BASE_DIR, 'datasets')


class DatasetHDF5(object):
    """
    parameters:
    filename: .hdf5 file
    """

    def __init__(self, filename, batch_size=32, npoints=1024, dim=5, shuffle=True, train=True):
        self.h5_files = filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.dim = dim
        self.train = train

        self.current_data = None
        self.current_label = None
        self.current_feature = None
        self.current_feature_label = None
        self.batch_idx = 0
        self.feature_batch_idx = 0
        self.reset()

    def reset(self):
        """ reset order of h5 files """
        self.current_data = None
        self.current_label = None
        self.batch_idx = 0

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data[:, :, 1:4])
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data)
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        batch_data[:, :, 1:4] = jittered_data
        return provider.shuffle_points(batch_data)

    def oversampling(self):
        ros = RandomOverSampler(random_state=42)
        reshape_data = self.current_data.reshape(len(self.current_data), -1)
        return ros.fit_resample(reshape_data, self.current_label)

    def _load_data_file(self, filename):
        all_data = None
        
        for file in filename:
            if not all_data:
                all_data, all_label = provider.load_h5_other(file)
            else:
                data, label = provider.load_h5_other(file)
                all_data = all_data.append(data)
                all_label = all_label.append(label)
        self.current_data, self.current_label = np.array(all_data), np.array(all_label)
        print(self.current_data.shape)
        print(self.current_label)

        self.current_data = self.current_data[:, :, :self.dim]
        self.current_label = np.squeeze(self.current_label)

        # oversampling less sample class
        if self.train:
            self.current_data, self.current_label = self.oversampling()
            self.current_data = self.current_data.reshape(len(self.current_data), self.npoints, self.dim)
            unique, counts = np.unique(self.current_label, return_counts=True)
            class_count = dict(zip(unique, counts))
            print(f'oversampled class: {class_count}')
        self.batch_idx = 0
        if self.shuffle:
            self.current_data, self.current_label, _ = provider.shuffle_data_other(
                self.current_data, self.current_label)

    def _has_next_batch_in_file(self):
        return self.batch_idx * self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return self.dim

    def has_next_batch(self):
        if self.current_data is None:
            self._load_data_file(self.h5_files)
            self.batch_idx = 0
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False):
        """ returned dimension may be smaller than self.batch_size """
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self.current_data.shape[0])

        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        if augment:
            data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch

    # functions for features
    def set_feature(self, global_feature, labels):
        self.current_feature = global_feature
        self.current_feature_label = labels

    def has_next_feature(self):
        return self.feature_batch_idx * self.batch_size < self.current_feature.shape[0]

    def next_feature(self):
        """ returned dimension may be smaller than self.batch_size """
        start_idx = self.feature_batch_idx * self.batch_size
        end_idx = min((self.feature_batch_idx + 1) * self.batch_size, self.current_feature.shape[0])

        feature_batch = self.current_feature[start_idx:end_idx, :].copy()
        feature_label_batch = self.current_feature_label[start_idx:end_idx].copy()
        self.feature_batch_idx += 1
        return feature_batch, feature_label_batch

    def reset_feature(self):
        """ reset order of h5 files """
        if self.shuffle:
            idx = np.arange(len(self.current_feature))
            np.random.shuffle(idx)
            self.current_feature = self.current_feature[idx, ...]
            self.current_feature_label = self.current_feature_label[idx, ...]
        self.feature_batch_idx = 0


if __name__ == '__main__':
    d = DatasetHDF5(os.path.join(DATADIR, 'testdataset_163_1024_dim6_normal.hdf5'), dim=6)
    # print(d.shuffle)
    # print(d.has_next_batch())
    while d.has_next_batch():
        ps_batch, cls_batch = d.next_batch()
        print(ps_batch.shape)
        print(cls_batch.shape)
