# program that saves the landmarks of the hand in a numpy array along with their images
# press R to start saving, Q to exit the program

import mediapipe as mp
import cv2
import numpy as np
import os

subdir = 'hand_zero'               # specify class subdirectory
n_samples_save = 100                # specify how many samples per category to save
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
iteration_counter = n_samples_save + 1
folder_counter = 1
X, y = [], []
mapping = {
    'hand_closed': 0,
    'hand_three': 1,
    'hand_open': 2,
    'hand_zero': 3,
}

hands = mp_hands.Hands(min_detection_confidence=0.2, static_image_mode=True)
capture = cv2.VideoCapture(0)  # 0 integrated | 1 plugged
while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_image = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    one_sample = []

    if detected_image.multi_hand_landmarks:
        for hand_lms in detected_image.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_lms,
                                      mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(255, 0, 255), thickness=4, circle_radius=2),
                                      connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(20, 180, 90), thickness=2, circle_radius=2)
                                      )

    cv2.imshow('Dataset Maker', image)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        iteration_counter = 1

    if iteration_counter < n_samples_save + 1:
        cv2.imwrite(os.path.join('data', subdir, f'{subdir}_image{iteration_counter}.jpg'), image)
        if detected_image.multi_hand_landmarks:
            for hand_lms in detected_image.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    one_sample.extend([lm.x, lm.y])
                X.append(one_sample)
                y.append(mapping[subdir])
        if iteration_counter == n_samples_save:
            print(f'Images for category {subdir} saved.')
        iteration_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

np.savez(os.path.join('data', f'data_{subdir}.npz'), X=X, y=y)
