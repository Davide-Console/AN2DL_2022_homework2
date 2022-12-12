import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

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


def fit_scaler(scaler_filename, data):
    scaler = MinMaxScaler()
    bidim_data = reshape(data)  # data.reshape(dims[0], -1)
    scaler = scaler.fit(bidim_data)
    joblib.dump(scaler, scaler_filename)


def apply_scaler(scaler_filename, data):
    scaler = joblib.load(scaler_filename)
    bidim_data = reshape(data)
    bidim_data = scaler.transform(bidim_data)
    data = restore_shape(bidim_data, data.shape)
    return data


def build_sequences(x_data, y_data, window, stride):
    x_output = []
    y_output = []
    dims = x_data.shape
    assert window % stride == 0
    for sample in range(dims[0]):
        padding_len = window - dims[1] % window
        padding = np.zeros((padding_len, dims[2]), dtype='float64')
        temp = np.concatenate((x_data[0, :, :], padding))
        idx = 0
        while idx + window <= dims[1]:
            x_output.append(temp[idx:idx + window])
            y_output.append(y_data[sample])
            idx += stride
    x_output = np.array(x_output)
    y_output = np.array(y_output)
    return x_output, y_output


def plot_sample(data):
    plt.plot(np.arange(0, data.shape[1]), data[0, :, 0])
    plt.show()


if __name__ == '__main__':
    x_data, y_data = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(x_data, y_data, split=0.7, shuffle=False)
    print('Before windowing')
    print('dataset dimensions: ', x_data.shape)
    print('labels length: ', y_data.shape)
    print('\n')
    print('total samples:\t', x_data.shape[0])
    print('training samples:\t', x_train.shape[0])
    print('validation samples:\t', x_test.shape[0])
    data, counts_data = np.unique(y_data, return_counts=True)
    train, counts_train = np.unique(y_train, return_counts=True)
    test, counts_test = np.unique(y_test, return_counts=True)
    print('\n')
    for i in range(12):
        print('class: ', i, '\tsamples: ', counts_data[i], '\tsplit: ', counts_train[i] / counts_data[i])
    plot_sample(x_train)
    plot_sample(x_test)

    fit_scaler('scaler.pkl', x_train)
    x_train = apply_scaler('scaler.pkl', x_train)
    x_test = apply_scaler('scaler.pkl', x_test)

    # x_data, y_data = build_sequences(x_data, y_data, 12, 3)
    # x_train, y_train = build_sequences(x_train, y_train, 12, 3)
    # x_test, y_test = build_sequences(x_test, y_test, 12, 3)

    print('After windowing')
    print('dataset dimensions: ', x_data.shape)
    print('labels length: ', y_data.shape)
    print('\n')
    print('total samples:\t', x_train.shape[0] + x_test.shape[0])
    print('training samples:\t', x_train.shape[0])
    print('validation samples:\t', x_test.shape[0])
    train, counts_train = np.unique(y_train, return_counts=True)
    test, counts_test = np.unique(y_test, return_counts=True)
    print('\n')
    for i in range(12):
        print('class: ', i, '\tsamples: ', counts_train[i] + counts_test[i], '\tsplit: ',
              counts_train[i] / (counts_train[i] + counts_test[i]))
    plot_sample(x_train)
    plot_sample(x_test)
