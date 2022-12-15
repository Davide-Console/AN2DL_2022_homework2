import os
import tensorflow as tf
import pickle
import numpy as np


class model:
    def __init__(self, path):
        self.scaler = pickle.load(open(os.path.join(path, 'SubmissionModel/scaler.pkl'), 'rb'))
        self.model0 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.7140-0.9274-f_model.h5'))
        self.model1 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6831-0.7488-f_model.h5'))
        self.model2 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.7058-0.8281-f_model.h5'))
        self.model3 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6605-0.7298-f_model.h5'))

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

    def predict(self, X):

        # Insert your preprocessing here
        bidim_data = self.reshape(X)
        bidim_data = self.scaler.transform(bidim_data)
        X = self.restore_shape(bidim_data, X.shape)

        out0 = self.model0.predict(X)
        out1 = self.model1.predict(X)
        out2 = self.model2.predict(X)
        X = self.reshape22D(X)
        out3 = self.model3.predict(X)

        out = out0 + out1 + out2 + out3
        out = tf.argmax(out, axis=-1)

        return out