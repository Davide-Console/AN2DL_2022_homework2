import os
import tensorflow as tf
import pickle
import numpy as np
from scipy.fft import fft

class model:
    def __init__(self, path):
        self.scaler = pickle.load(open(os.path.join(path, 'SubmissionModel/scaler.pkl'), 'rb'))
        self.model0 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.7140-0.9274-f_model.h5'))
        self.model1 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6605-0.7298-f_model.h5'))
        self.model2 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6955-0.8456-f_model.h5'))
        self.model3 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6914-0.8981-f_model.h5'))
        #self.model4 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6831-0.7488-f_model.h5'))
        #self.model5 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.7058-0.8281-f_model.h5'))

    def reshape(self, data):
        dims = data.shape
        reshaped_data = np.zeros((dims[0] * dims[1], dims[2]))
        element = 0
        for sample in range(dims[0]):
            for timestep in range(dims[1]):
                reshaped_data[element] = data[sample, timestep, :]
                element += 1
        return reshaped_data

    def restore_shape(self, reshaped_data, original_shape):
        restored_shape = (original_shape[0], original_shape[1], original_shape[2])
        restored_data = np.zeros(restored_shape)
        element = 0
        for sample in range(original_shape[0]):
            for timestep in range(original_shape[1]):
                restored_data[sample, timestep, :] = reshaped_data[element]
                element += 1
        return restored_data

    def feature_to_2D(self, feature):
        dims = feature.shape
        feature_to_img = np.zeros((dims[0], dims[0]))
        for i in range(dims[0]):
            feature_to_img[i, :] = feature
        for j in range(dims[0]):
            feature_to_img[:, j] = feature_to_img[:, j] - feature

        return feature_to_img

    def reshape22D(self, data):
        dims = data.shape
        reshaped_data = np.zeros((dims[0], dims[1], dims[1], dims[2]))
        for sample in range(dims[0]):
            signal_to_img = np.zeros((dims[1], dims[1], dims[2]))
            for feature in range(dims[2]):
                signal_to_img[:, :, feature] = self.feature_to_2D(data[sample, :, feature])
            reshaped_data[sample] = signal_to_img
        return reshaped_data

    def add_fft(self, data):
        dims = data.shape
        data_and_fft = np.zeros((dims[0], dims[1], dims[2] * 2))

        for sample in range(dims[0]):
            for feature in range(dims[2]):
                data_and_fft[sample, :, feature] = data[sample, :, feature]
                data_and_fft[sample, :, feature + dims[2]] = abs(fft(data[sample, :, feature]))

        return data_and_fft

    def predict(self, X):

        # Insert your preprocessing here
        bidim_data = self.reshape(X)
        bidim_data = self.scaler.transform(bidim_data)
        X = self.restore_shape(bidim_data, X.shape)

        #out0 = self.model0.predict(X)
        X1 = self.reshape22D(X)
        out1 = self.model1.predict(X1)
        X2 = self.add_fft(X)
        out2 = self.model2.predict(X2)
        X3 = X[:, :, [0, 1, 2, 3, 5]]
        out3 = self.model3.predict(X3)
        #out5 = self.model1.predict(X)
        #out4 = self.model2.predict(X)

        out = out3 + out1 + out2 #+ out3
        out = tf.argmax(out, axis=-1)

        return out