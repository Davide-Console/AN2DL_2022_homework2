from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from data_utils import *

seed = 1
import tensorflow as tf
from submission.model import model
tfk=tf.keras
tfkl=tf.keras.layers

x_data, y_data = load_dataset()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
    x_train = x_data[train_index]
    y_train = y_data[train_index]
    x_test = x_data[test_index]
    y_test = y_data[test_index]

y_train = tfk.utils.to_categorical(y_train)
y_test = tfk.utils.to_categorical(y_test)

# declare model

model = model('submission')
y_pred = model.predict(x_test)
y_true = y_test

scores = classification_report(np.argmax(y_true, axis=-1), y_pred, digits=4,
                               output_dict=True)

print(classification_report(np.argmax(y_true, axis=-1), y_pred, digits=4,
                            output_dict=False))