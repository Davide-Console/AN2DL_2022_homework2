import os
import tensorflow as tf
import pickle
import numpy as np

class model:
    def __init__(self, path):
        self.scaler = pickle.load(open(os.path.join(path, 'SubmissionModel/scaler.pkl'), 'rb'))
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.7058-0.8281-f_model.h5'))

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

    def predict(self, X):
        
        # Insert your preprocessing here
        bidim_data = self.reshape(X)
        bidim_data = self.scaler.transform(bidim_data)
        X = self.restore_shape(bidim_data, X.shape)

        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out
