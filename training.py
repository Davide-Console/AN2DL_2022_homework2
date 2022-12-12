from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from data_utils import *
from networks import *
import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
import random
import execution_settings

# Random seed for reproducibility
seed = 313

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


def get_callbacks():
    tboard = 'tb_logs'
    os.makedirs(tboard, exist_ok=True)
    tb_call = TensorBoard(log_dir=tboard)

    chkpt_dir = 'float_model'
    os.makedirs(chkpt_dir, exist_ok=True)
    chkpt_call = ModelCheckpoint(
        filepath=os.path.join(chkpt_dir, '{val_accuracy:.4f}-{accuracy:.4f}-f_model.h5'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True)

    logdir = 'train_log.csv'
    csv_logger = CSVLogger(logdir, append=True, separator=';')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.05, patience=100,
                                                      restore_best_weights=False)
    reduce_on_plateau = tfk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.5, min_lr=1e-5)

    return [tb_call, chkpt_call, csv_logger, early_stopping, reduce_on_plateau]


def train():
    # load dataset
    x_data, y_data = load_dataset()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
        x_train = x_data[train_index]
        y_train = y_data[train_index]
        x_test = x_data[test_index]
        y_test = y_data[test_index]
    #
    # normalize dataset
    fit_scaler('scaler.pkl', x_train)
    x_train = apply_scaler('scaler.pkl', x_train)
    x_test = apply_scaler('scaler.pkl', x_test)
    #
    # # windowing
    # x_train, y_train = build_sequences(x_train, y_train, 12, 3)
    # x_test, y_test = build_sequences(x_test, y_test, 12, 3)

    print(x_train.shape[0])
    print(y_train.shape[0])
    y_train = tfk.utils.to_categorical(y_train)
    y_test = tfk.utils.to_categorical(y_test)

    # declare model
    batch_size = 128
    filters = 128

    model = build_1DCNN_classifier(x_train.shape[1:], y_train.shape[-1], filters=filters)
    #model = build_LSTM_classifier(x_train.shape[1:], y_train.shape[-1], filters=filters)
    #model = build_BiLSTM_classifier(x_train.shape[1:], y_train.shape[-1], filters=filters)

    model.summary()

    # training
    epochs = 500
    callbacks = get_callbacks()

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    y_true = y_test
    y_pred = model.predict(x_test)

    print(classification_report(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), digits=4,
                                output_dict=False))


if __name__ == '__main__':
    train()
