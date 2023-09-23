# also try with ordinal encoding / no encoding + SparseCategoricalCrossentropy
# also try using padding = 'same' for ConvLSTM2D
# something is wrong with the model

import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import datetime as dt
import matplotlib.pyplot as plt
from moviepy.editor import *
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import *
from tensorflow.keras.layers import *

seed = 77
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
plt.figure(figsize=(10, 10))

data_parent_dir = 'UCF50'
image_height, image_width = 64, 64
sequence_length = 20
category_list = ['Nunchucks', 'JugglingBalls', 'PizzaTossing', 'PullUps']

# all_categories_list = os.listdir(all_data_dir)
# for category_name in all_categories_list:
#     current_category_subdir_path = os.path.join(all_data_dir, category_name)
#     video_file_names = os.listdir(current_category_subdir_path)
#     chosen_video_file_name = random.choice(video_file_names)
#     video_reader = cv2.VideoCapture(os.path.join(current_category_subdir_path, chosen_video_file_name))
#     ret, frame_bgr = video_reader.read()
#     video_reader.release()
#     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#     cv2.putText(frame_rgb, category_name, (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 255, 255))
#
#     plt.imshow(frame_rgb)
#     plt.show()

def frames_preprocessing(video_path):
    """
    Resizes, normalizaes, and then extracts frames from a video file.
    :param video_path: String that indicates the path of the video file.
    :return: A list with resized and normalized frames.
    """

    frames_list = []
    cap = cv2.VideoCapture(video_path)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip_length = max(int(total_frame_count / sequence_length), 1)

    for frame_idx in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_skip_length * frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255.
        frames_list.append(normalized_frame)

    cap.release()
    return frames_list

def create_dataset(class_list):
    """
    Extracts the frames from all video files of the selected classes and makes a dataset.
    :param class_list: A list of class names whose video files need to be extracted.
    :return X: A list containing the lists of extracted frames of a video file.
    :return y: A list of class indices of the video files.
    :return video_file_paths: A list containing the paths of the video files.
    """

    X = []
    y = []
    video_file_paths = []

    for idx, class_ in enumerate(class_list):
        print(f'Now processing - {class_}')
        current_class_path = os.path.join(data_parent_dir, class_)
        class_all_video_names = os.listdir(current_class_path)

        for video_name in class_all_video_names:
            video_path = os.path.join(current_class_path, video_name)
            frames = frames_preprocessing(video_path)

            if len(frames) == sequence_length:
                X.append(frames)
                y.append(idx)
                video_file_paths.append(video_path)

    X = np.array(X) # shape is (n_videos, n_frames, height, width, channels)
    y = np.array(y) # shape is (n_videos)

    return X, y, video_file_paths

X, y, video_paths = create_dataset(category_list)
y = tf.keras.utils.to_categorical(y) # one hot encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)

early_stopping = tf.keras.callbacks.EarlyStopping(restore_best_weights=True,
                                                  patience=20)

model = tf.keras.Sequential([
    ConvLSTM2D(8, 3, activation='tanh', input_shape=(sequence_length, image_height, image_width, 3),
               return_sequences=True, data_format='channels_last', recurrent_dropout=0.3),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    TimeDistributed(Dropout(0.3)),

    ConvLSTM2D(8, 3, activation='tanh', return_sequences=True,
               data_format='channels_last', recurrent_dropout=0.3),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    TimeDistributed(Dropout(0.3)),

    ConvLSTM2D(16, 3, activation='tanh', return_sequences=True,
               data_format='channels_last', recurrent_dropout=0.3),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    TimeDistributed(Dropout(0.3)),

    ConvLSTM2D(20, 3, activation='tanh', return_sequences=True,
               data_format='channels_last', recurrent_dropout=0.3),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    TimeDistributed(Dropout(0.3)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(category_list)),
])

print(model.summary())

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'],
)

model_train_hist = model.fit(
    X_train, y_train,
    shuffle=True,
    batch_size=8,
    epochs=70,
    validation_split=0.2,
    callbacks=[early_stopping],
)

model_eval_loss, model_eval_acc = model.evaluate(X_test, y_test)
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_str = dt.datetime.strftime(current_date_time_dt, date_time_format)

model_name = f'model__date_time_{current_date_time_str}__loss_{model_eval_loss}__acc_{model_eval_acc}_ConvLSTM2D_Nunchucks_JugglingBalls_PizzaTossing_PullUps.h5'
model.save(model_name)

df_train_hist = pd.DataFrame(model_train_hist.history)
df_train_hist.loc[:, ['loss', 'val_loss']].plot()