"""
    Model to read a list of .hdf5 files

"""

import os
import sys
import numpy as np
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
DATADIR = os.path.join(BASE_DIR, 'datasets')


class DatasetHDF5(object):
    """
    parameters:
    list_filename: list of .hdf5 files
    """
    def __init__(self, list_filename, batch_size=32, npoints=1024, dim=5, shuffle=True):
        self.h5_files = list_filename
        print(self.h5_files)
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.dim = dim

        self.file_idxs = np.arange(0, len(self.h5_files))
        self.current_data = None
        self.current_label = None
        self.current_feature = None
        self.current_feature_label = None
        self.current_file_idx = 0
        self.batch_idx = 0
        self.feature_batch_idx = 0
        self.reset()

    def reset(self):
        """ reset order of h5 files """
        if self.shuffle:
            np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data)
        # rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        # jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        # jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(rotated_data)
        # rotated_data[:, :, 0:3] = jittered_data
        return jittered_data  # provider.shuffle_points(jittered_data)

    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        # print(filename)
        self.current_data, self.current_label = provider.load_h5_other(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0
        if self.shuffle:
            self.current_data, self.current_label, _ = provider.shuffle_data_other(
                self.current_data, self.current_label)

    def _has_next_batch_in_file(self):
        return self.batch_idx * self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return self.dim

    def has_next_batch(self):
        if (self.current_data is None) or (not self._has_next_batch_in_file()):
            if self.current_file_idx >= len(self.h5_files):
                return False
            self._load_data_file(self._get_data_filename())
            self.batch_idx = 0
            self.current_file_idx += 1
            # get weight for each class
            self.weight_dict = provider.weight_dict_fc(self.current_label)
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
    d = DatasetHDF5([os.path.join(DATADIR,'testdataset_154_1024_dim5.hdf5')])
    # print(d.shuffle)
    # print(d.has_next_batch())
    while d.has_next_batch():
        ps_batch, oth_batch, cls_batch = d.next_batch(True)
        print(ps_batch.shape)
        print(oth_batch.shape)
        print(cls_batch.shape)
