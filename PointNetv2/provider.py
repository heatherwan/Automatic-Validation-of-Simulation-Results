import h5py
import numpy as np
from sklearn.preprocessing import label_binarize

from Parameters import Parameters

para = Parameters()
h5py.get_config().default_file_mode = 'r'


def shuffle_data_other(data, other, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          other: B,additional features,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, other, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], other[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def load_h5_other(h5_filename):
    f = h5py.File(h5_filename)
    data = f["data"][:]
    other = f["other"][:]
    label = f["label"][:]
    return data, other, label


def loadDataFile_other(filename):
    return load_h5_other(filename)


# for segmentation
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f["data"][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return data, label, seg


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


def weight_dict_fc(trainLabel):
    y_total = label_binarize(trainLabel, classes=[i for i in range(para.outputClassN)])
    class_distribution_class = np.sum(y_total, axis=0)  # get count for each class
    class_distribution_class = [float(i) for i in class_distribution_class]
    class_distribution_class = class_distribution_class / np.sum(class_distribution_class)  # get ratio for each class
    inverse_dist = 1 / class_distribution_class
    norm_inv_dist = inverse_dist / np.sum(inverse_dist)
    weights = norm_inv_dist * para.weight_scaler + 1  # scalar should be reconsider
    weight_dict = dict()
    for classID, value in enumerate(weights):
        weight_dict.update({classID: value})
    return weight_dict


def weights_calculation(batch_labels, weight_dict):
    weights = []
    for i in batch_labels:
        weights.append(weight_dict[i])
    return weights
