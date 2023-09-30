# Author: Odilbek Tokhirov
# loads the saved data and trains the Dense Neural Network model

import pandas as pd
import numpy as np
import tensorflow as tf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import *
from tensorflow.keras.layers import *

# loading data
class_names = ['hand_closed', 'hand_three', 'hand_open', 'hand_zero']
loaded_data = np.load(fr'data\data_{class_names[0]}.npz')
X, y = loaded_data['X'], loaded_data['y']
for data_class in class_names[1:]:
    loaded_data = np.load(fr'data\data_{data_class}.npz')
    X = np.vstack((X, loaded_data['X']))
    y = np.concatenate((y, loaded_data['y']))

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
)

with tf.device('/GPU:0'):
    model = tf.keras.Sequential([
        Dense(16, activation='relu', input_shape=(42,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(4),
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
    )

    model_train_hist = model.fit(
        X_train, y_train,
        shuffle=True,
        batch_size=50,
        epochs=70,
        validation_split=0.25,
        callbacks=[early_stopping],
    )
    print(model.summary())

    model_eval_loss, model_eval_acc = model.evaluate(X_test, y_test)
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_str = dt.datetime.strftime(current_date_time_dt, date_time_format)

    model_name = f'model_nn__date_time_{current_date_time_str}__loss_{model_eval_loss}__acc_{model_eval_acc}__hand__oneimage.h5'
    model.save(model_name)

    df_train_hist = pd.DataFrame(model_train_hist.history)
    df_train_hist.loc[:, ['loss', 'val_loss']].plot()
    plt.show()
