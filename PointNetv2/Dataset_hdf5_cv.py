"""
    Model to read a list of .hdf5 files

"""

import os
import sys
import numpy as np
import provider
from sklearn.model_selection import KFold


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
DATADIR = os.path.join(BASE_DIR, 'datasets')


class DatasetHDF5_Kfold(object):
    """
    parameters:
    list_filename: list of .hdf5 files
    """

    def __init__(self, filename, batch_size=13, npoints=1024, dim=5, shuffle=True, cv=5):
        self.h5_file = filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.dim = dim
        self.cv = cv

        self.current_data = None
        self.current_label = None
        self.batch_idx = 0

        self.init_data(self.h5_file)
        self.trainvalid_index = list(KFold(self.cv).split(self.current_data))
        self.train_data = None
        self.valid_data = None
        self.train_label = None
        self.valid_label = None

    def init_data(self, filename):
        self.current_data, self.current_label = provider.load_h5_other(filename)
        self.current_label = np.squeeze(self.current_label)

    def set_data(self, split):
        self.train_data = self.current_data[self.trainvalid_index[split][0]]
        self.valid_data = self.current_data[self.trainvalid_index[split][1]]
        self.train_label = self.current_label[self.trainvalid_index[split][0]]
        self.valid_label = self.current_label[self.trainvalid_index[split][1]]

    def reset(self):
        if self.shuffle:
            self.current_data, self.current_label, _ = provider.shuffle_data_other(
                self.current_data, self.current_label)
        self.batch_idx = 0

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data[:, :, :3])
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data)
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        batch_data[:, :, :3] = jittered_data
        return provider.shuffle_points(batch_data)

    def _has_next_batch_in_file(self):
        return self.batch_idx * self.batch_size < self.train_data.shape[0]

    def num_channel(self):
        return self.dim

    def has_next_batch(self):
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False, train=True):
        """ returned dimension may be smaller than self.batch_size """
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self.current_data.shape[0])
        if train:
            data_batch = self.train_data[start_idx:end_idx, 0:self.npoints, :].copy()
            label_batch = self.train_label[start_idx:end_idx].copy()
        else:
            data_batch = self.valid_data[start_idx:end_idx, 0:self.npoints, :].copy()
            label_batch = self.valid_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        if augment:
            data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch


if __name__ == "__main__":
    INPUT_PATH = os.path.join(DATADIR, 'traindataset_651_1024_dim6_SF10.hdf5')
    DatasetHDF5_Kfold(INPUT_PATH)