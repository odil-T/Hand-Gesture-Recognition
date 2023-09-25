# Hand-Gesture-Recognition

This repository stores the source code for the hand gesture recognition model.

When the 'hand_gesture_reader.py' file is executed, a webcam window opens which can predict four gestures. The program then activates certain arrow keys that are mapped to certain gestures.
The sample images of gestures can be found in this repository.

The above files uses the random forest model parameters from 'model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl' file. These parameters were trained in 'model_hand_rf.py' file using the data stored in .npz files.

The data was made using 'hand landmark dataset maker.py' file.
