# Hand-Gesture-Recognition

This repository stores the source code for the hand gesture recognition model.

When the `hand_gesture_reader.py` file is executed, a webcam window opens which can predict four gestures. The program then activates certain arrow keys that are mapped to certain gestures.
The sample images of gestures can be found at the end of README.

The above files uses the random forest model parameters from `model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl` file. These parameters were trained in `model_hand_rf.py` file using the data stored in .npz files.

The data was made using `hand_landmark_dataset_maker.py` file.


Hand Zero
![hand_zero_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/ad40f4fc-0d5c-42c2-af1c-96e83f17b630)


Hand Closed
![hand_closed_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/8c2401aa-a8a5-4ed0-9240-ae662802468c)


Hand Open
![hand_open_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/eb28fa08-d13a-4ec6-bfbf-7f38b380cbf3)


Hand Three
![hand_three_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/f8c36198-6003-4794-9e01-bf4bfe40f43d)
