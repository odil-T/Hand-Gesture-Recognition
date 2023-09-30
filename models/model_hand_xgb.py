# Author: Odilbek Tokhirov
# loads the saved data and trains the XGBoost model for single image classification

import pandas as pd
import xgboost as xgb
import numpy as np
import tensorflow as tf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# loading data
class_names = ['hand_closed', 'hand_three', 'hand_open', 'hand_four']
loaded_data = np.load(fr'data\data_{class_names[0]}.npz')
X, y = loaded_data['X'], loaded_data['y']
for data_class in class_names[1:]:
    loaded_data = np.load(fr'data\data_{data_class}.npz')
    X = np.vstack((X, loaded_data['X']))
    y = np.concatenate((y, loaded_data['y']))

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=1, stratify=y)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, shuffle=True, test_size=0.25, random_state=1, stratify=y_train)

with tf.device('/GPU:0'):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    yhat_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, yhat_test)
    print(f'Accuracy: {test_accuracy}')

    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_str = dt.datetime.strftime(current_date_time_dt, date_time_format)

    model_name = f'model_xgb__date_time_{current_date_time_str}__acc_{test_accuracy}__hand__oneimage.pkl'
    joblib.dump(model, model_name)
