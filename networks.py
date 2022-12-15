import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.layers import Input, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, model_from_json
import os
import tempfile
from tensorflow.keras import backend as K

from tensorflow.python.keras.applications.efficientnet import EfficientNetB0

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


def get_EfficientNetB0(weights=None, input_shape=(36, 36, 6), classes=12, regularize=True, l1=0.0001, l2=0.00001):
    model = EfficientNetB0(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def cbr(input, filters, kernel_size, strides):
    '''
    Convolution - BatchNorm - ReLU - Dropout
    '''
    net = layers.Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer='he_uniform',
                        kernel_regularizer=regularizers.l2(0.01),
                        strides=strides, padding='same')(input)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    net = Dropout(rate=0.2)(net)
    return net


def skip_blk(input, filters, kernel_size=3, strides=1):
    net = cbr(input=input, filters=filters, kernel_size=kernel_size, strides=strides)
    net = cbr(input=net, filters=filters, kernel_size=kernel_size, strides=strides)
    net = cbr(input=net, filters=filters, kernel_size=kernel_size, strides=strides)
    skip = cbr(input=input, filters=filters, kernel_size=kernel_size, strides=strides)
    net = layers.Add()([skip, net])
    net = cbr(input=net, filters=filters, kernel_size=3, strides=strides * 2)
    return net


def customcnn(input_shape=(36, 36, 6), classes=12, filters=None):
    '''
    Arguments:
      input_shape: tuple of integers indicating height, width, channels
      classes    : integer to set number of classes
      filters    : list of integers - each list element sets the number of filters used in a skip block
    '''

    if filters is None:
        filters = [8, 16, 32, 64, 128]
    input_layer = Input(shape=input_shape)
    net = input_layer

    for f in filters:
        net = skip_blk(net, f)

    # reduce channels
    f = int((filters[-1] / 2 - classes) / 2)
    net = skip_blk(net, f)

    # create a conv layer that will reduce feature maps to (1,1,classes)
    h = K.int_shape(net)[1]
    w = K.int_shape(net)[2]

    net = layers.Conv2D(filters=classes, kernel_size=(h, w), kernel_initializer='he_uniform',
                        kernel_regularizer=regularizers.l2(0.01), strides=w, padding='valid')(net)

    net = layers.Flatten()(net)

    output_layer = layers.Activation('softmax')(net)

    return Model(inputs=input_layer, outputs=output_layer)

if __name__ == '__main__':
    model = customcnn()
    print(model.summary())
