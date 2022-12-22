import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler

np.random.seed(313)


def load_dataset():
    x_path = 'dataset/x_train.npy'
    y_path = 'dataset/y_train.npy'
    x_data = np.load(x_path)
    y_data = np.load(y_path)
    return x_data, y_data


def split_dataset(x_data, y_data, split=0.8, shuffle=False):
    x_train = []
    y_train = []
    x_validation = []
    y_validation = []
    classes, classes_index = np.unique(y_data, return_index=True)

    for label in range(12):
        start = classes_index[label]
        if label < 11:
            len_class = classes_index[label + 1] - classes_index[label]
        else:
            len_class = y_data.shape[0] - classes_index[label]

        samples_number = math.ceil(len_class * split)

        for i in range(samples_number):
            x_train.append(x_data[start + i, :, :])
            y_train.append(y_data[start + i])
        for i in range(len_class - samples_number):
            x_validation.append(x_data[start + samples_number + i, :, :])
            y_validation.append(y_data[start + samples_number + i])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    if shuffle is True:
        np.random.shuffle(x_train)
        np.random.shuffle(y_train)
        np.random.shuffle(x_validation)
        np.random.shuffle(y_validation)

    return x_train, x_validation, y_train, y_validation


def reshape(data):
    dims = data.shape
    reshaped_data = np.zeros((dims[0] * dims[1], dims[2]))
    element = 0
    for sample in range(dims[0]):
        for timestep in range(dims[1]):
            reshaped_data[element] = data[sample, timestep, :]
            element += 1
    return reshaped_data


def restore_shape(reshaped_data, original_shape):
    restored_shape = (original_shape[0], original_shape[1], original_shape[2])
    restored_data = np.zeros(restored_shape)
    element = 0
    for sample in range(original_shape[0]):
        for timestep in range(original_shape[1]):
            restored_data[sample, timestep, :] = reshaped_data[element]
            element += 1
    return restored_data


def add_fft(data):
    dims = data.shape
    data_and_fft = np.zeros((dims[0], dims[1], dims[2]*2))

    for sample in range(dims[0]):
        for feature in range(dims[2]):
            data_and_fft[sample, :, feature] = data[sample, :, feature]
            data_and_fft[sample, :, feature+dims[2]] = abs(fft(data[sample, :, feature]))

    return data_and_fft

def fit_scaler(scaler_filename, data):
    scaler = StandardScaler()
    bidim_data = reshape(data)
    scaler = scaler.fit(bidim_data)
    pickle.dump(scaler, open(scaler_filename, 'wb'))


def apply_scaler(scaler_filename, data):
    scaler = pickle.load(open(scaler_filename, 'rb'))
    bidim_data = reshape(data)
    bidim_data = scaler.transform(bidim_data)
    data = restore_shape(bidim_data, data.shape)
    return data


def feature_to_2D(feature):
    dims = feature.shape
    feature_to_img = np.zeros((dims[0], dims[0]))
    for i in range(dims[0]):
        feature_to_img[i, :] = feature
    for j in range(dims[0]):
        feature_to_img[:, j] = feature_to_img[:, j] - feature

    return feature_to_img


def reshape22D(data):
    dims = data.shape
    reshaped_data = np.zeros((dims[0], dims[1], dims[1], dims[2]))
    for sample in range(dims[0]):
        signal_to_img = np.zeros((dims[1], dims[1], dims[2]))
        for feature in range(dims[2]):
            signal_to_img[:, :, feature] = feature_to_2D(data[sample, :, feature])
        reshaped_data[sample] = signal_to_img
    return reshaped_data


def build_sequences(x_data, y_data, window, stride):
    x_output = []
    y_output = []
    dims = x_data.shape
    assert window % stride == 0
    for sample in range(dims[0]):
        padding_len = window - dims[1] % window
        padding = np.zeros((padding_len, dims[2]), dtype='float64')
        temp = np.concatenate((x_data[sample, :, :], padding))
        idx = 0
        while idx + window <= len(temp):
            x_output.append(temp[idx:idx + window])
            y_output.append(y_data[sample])
            idx += stride
    x_output = np.array(x_output)
    y_output = np.array(y_output)
    return x_output, y_output


def plot_sample(data):
    plt.plot(np.arange(0, data.shape[1]), data[0, :, 0])
    plt.show()


def add_fft(data):
    dims = data.shape
    data_and_fft = np.zeros((dims[0], dims[1], dims[2]*2))

    for sample in range(dims[0]):
        for feature in range(dims[2]):
            data_and_fft[sample, :, feature] = data[sample, :, feature]
            data_and_fft[sample, :, feature+dims[2]] = abs(fft(data[sample, :, feature]))

    return data_and_fft
