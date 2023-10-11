# Author: Odilbek Tokhirov
# run the following command in terminal:
# streamlit run hand_gesture_reader_deployed.py

import mediapipe as mp
import cv2
import numpy as np
import joblib
import pyautogui as pag
import streamlit as st
import os

# initialization
model_name_rf = 'model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl'
model_name_xgb = 'model_XGBoost__date_time_2023_09_23__11_55_17__acc_1.0__hand__oneimage.pkl'
model_name_nn = ''
model = joblib.load(model_name_rf)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
idx_to_class = {
    0: 'Closed',
    1: 'Three',
    2: 'Open',
    3: 'Zero',
}
class_to_key = {
    'Closed': 'up',
    'Three': 'right',
    'Open': 'left',
    'Zero': 'down',
}
n_classes = len(class_to_key)

# streamlit code
st.title('Hand Gesture Reader')
frame_placeholder = st.empty()
st.subheader('Predicted Probabilities:')
text_boxes = [col.empty() for col in st.columns(n_classes)]

st.subheader('Activated Keys:')
key_inputs = []
for key, col in zip(class_to_key, st.columns(n_classes)):
    with col:
        key_input = st.text_input(f'Hand {key}', class_to_key[key])
        key_inputs.append(key_input)

# description section
description = '''
This is a hand gesture recognition app. It uses the webcam to recognize hand gestures of the right hand and performs actions corresponding
to the gestures. When your hand is shown to the camera, the recognized gesture will be displayed and a certain key will
be activated. You can change the type of key that is activated. Just type the key name above in the gesture category to which
you wish to assign. Please not that the page will refresh after updating a key. You can refer to pyautogui documentation on how to write certain keys (for e.g. F1 key).
The sample gestures are given below:
'''
st.header('Description')
st.markdown(description)
for i in range(0, n_classes, 2):
    for col, j in zip(st.columns(2), range(2)):
        with col:
            image_name = os.listdir('sample_images')[i + j]
            st.image(os.path.join('sample_images', image_name), caption=image_name[:-4], width=320)

current_command = None
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

capture = cv2.VideoCapture(1)  # 0 integrated | 1 plugged
while capture.isOpened():
    ret, frame = capture.read()
    height, width = frame.shape[:-1]
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_image = hands.process(image)
    x = []

    if detected_image.multi_hand_landmarks:
        for hand_lms in detected_image.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_lms,
                                      mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(255, 0, 255), thickness=4, circle_radius=2),
                                      connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(20, 180, 90), thickness=2, circle_radius=2)
                                      )
            for lm in hand_lms.landmark:
                x.extend([lm.x, lm.y])

        x = np.array(x)  # 1D array of current landmark positions
        x_max = int(width * np.max(x[::2]))
        x_min = int(width * np.min(x[::2]))
        y_max = int(height * np.max(x[1::2]))
        y_min = int(height * np.min(x[1::2]))
        x = x[None, :]  # adds a new dimension to x to avoid input shape error
        predicted_class_probabilities = model.predict_proba(x)[0]

        # updates predicted probabilities of classes
        for text_box, probability, i in zip(text_boxes, predicted_class_probabilities, range(n_classes)):
            text_box.text(f'Hand {list(class_to_key.keys())[i]} = {probability}')

        if np.max(predicted_class_probabilities) > 0.5:  # threshold to predict only if >50% certain
            yhat_idx = np.argmax(predicted_class_probabilities)
            yhat = idx_to_class[yhat_idx]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)  # draws a rectangle and types the predicted class
            cv2.putText(image, f'{yhat}', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

            if current_command != yhat:  # performs the new action once
                pag.press(key_inputs[yhat_idx])
                current_command = yhat

    elif current_command != None:
        current_command = None

    frame_placeholder.image(image, channels='RGB')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
