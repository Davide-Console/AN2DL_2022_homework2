import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.layers import Input, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, model_from_json
from sklearn.model_selection import StratifiedShuffleSplit
from data_utils import *
import os
import tempfile

tfk = tf.keras
tfkl = tf.keras.layers
seed = 313


def attach_final_layers(model, classes):
    """
        This function takes in a model and the number of classes and returns the model with the final layers attached.
        The final layers are a Global Average Pooling layer, a Dropout layer and a Dense layer with the number of classes
        as the number of neurons.
        Parameters:
        model (keras.Model): The model to which the final layers are to be attached.
        classes (int): The number of classes in the dataset.
        Returns:
        keras.Model: The model with the final layers attached.
    """
    model = GlobalAvgPool2D()(model)
    model = Dropout(rate=0.4)(model)

    activation = 'softmax' if classes > 2 else 'sigmoid'
    classes = 1 if classes == 2 else classes

    output_layer = Dense(classes, activation=activation,
                         bias_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),
                         activity_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))(model)

    return output_layer


def add_regularization(model, l1, l2):
    """
        This function adds regularization to a model.
        Parameters
        ----------
        model : keras.models.Model
            The model to add regularization to.
        l1 : float
            The l1 regularization coefficient.
        l2 : float
            The l2 regularization coefficient.
        Returns
        -------
        keras.models.Model
            The model with regularization.
    """
    regularizer = regularizers.l1_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def build_1DCNN_classifier(input_shape, classes, filters=128):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Feature extractor
    cnn = tfkl.Conv1D(filters, 3, padding='same', activation='relu')(input_layer)
    cnn = tfkl.MaxPooling1D()(cnn)
    cnn = tfkl.Conv1D(filters, 3, padding='same', activation='relu')(input_layer)
    cnn = tfkl.MaxPooling1D()(cnn)
    cnn = tfkl.Conv1D(filters, 3, padding='same', activation='relu')(cnn)
    gap = tfkl.GlobalAveragePooling1D()(cnn)
    dropout = tfkl.Dropout(.5, seed=seed)(gap)

    # Classifier
    classifier = tfkl.Dense(filters, activation='relu')(dropout)
    output_layer = tfkl.Dense(classes, activation='softmax')(classifier)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Return the model
    return model


def build_LSTM_classifier(input_shape, classes, units=128):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Feature extractor
    lstm = tfkl.LSTM(units, return_sequences=True)(input_layer)
    lstm = tfkl.LSTM(units)(lstm)
    dropout = tfkl.Dropout(.5, seed=seed)(lstm)

    # Classifier
    classifier = tfkl.Dense(units, activation='relu')(dropout)
    output_layer = tfkl.Dense(classes, activation='softmax')(classifier)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Return the model
    return model


def build_BiLSTM_classifier(input_shape, classes, units):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Feature extractor
    bilstm = tfkl.Bidirectional(tfkl.LSTM(units, return_sequences=True))(input_layer)
    bilstm = tfkl.Bidirectional(tfkl.LSTM(units))(bilstm)
    dropout = tfkl.Dropout(.5, seed=seed)(bilstm)

    # Classifier
    classifier = tfkl.Dense(units, activation='relu')(dropout)
    output_layer = tfkl.Dense(classes, activation='softmax')(classifier)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Return the model
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        n_classes=12
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = tfk.Model(inputs, outputs, name='model')

    return model


def customcnn(input_shape, classes):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation='relu')(input_layer)
    x = tfkl.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = tfkl.MaxPooling1D(pool_size=2, strides=2)(x)
    cnn = tfkl.Conv1D(filters=256, kernel_size=3, padding="same", activation="relu")(x)
    cnn = tfkl.MaxPooling1D(pool_size=2, strides=2)(cnn)
    cnn = tfkl.Conv1D(filters=512, kernel_size=3, padding="same", activation="relu")(cnn)
    cnn = tfkl.GlobalAveragePooling1D()(cnn)
    cnn = tfkl.Dense(512, activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4))(cnn)
    cnn = tfkl.Dense(256, activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4))(cnn)
    cnn = tfkl.Dense(128, activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4))(cnn)
    cnn = tfkl.Dropout(0.4)(cnn)
    output_layer = tfkl.Dense(classes, activation="softmax",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4))(cnn)

    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Return the model
    return model


def build_NN_classifier(input_shape, classes, filters=128):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Classifier
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation='relu')(input_layer)
    x = tfkl.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = tfkl.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(filters=256, kernel_size=3, padding="same", activation="relu")(x)
    x = tfkl.MaxPooling1D(pool_size=2, strides=2)(x)
    cnn = tfkl.Conv1D(filters=512, kernel_size=3, padding="same", activation="relu")(x)
    cnn = tfkl.GlobalAveragePooling1D()(cnn)
    classifier = tfkl.Dense(512, activation='relu')(cnn)
    dropout = tfkl.Dropout(.5, seed=seed)(classifier)
    classifier = tfkl.Dense(256, activation='relu')(dropout)
    dropout = tfkl.Dropout(.5, seed=seed)(classifier)
    classifier = tfkl.Dense(128, activation='relu')(dropout)
    dropout = tfkl.Dropout(.5, seed=seed)(classifier)
    classifier = tfkl.Dense(64, activation='relu')(dropout)
    dropout = tfkl.Dropout(.5, seed=seed)(classifier)
    output_layer = tfkl.Dense(classes, activation='softmax')(dropout)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Return the model
    return model


if __name__ == '__main__':
    x_data, y_data = load_dataset()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
        x_train = x_data[train_index]
        y_train = y_data[train_index]
        x_test = x_data[test_index]
        y_test = y_data[test_index]

    model = build_FFNN_classifier(x_train.shape[1:], y_train.shape[-1], filters=128)
    model.summary()
