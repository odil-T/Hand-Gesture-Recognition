# Hand-Gesture-Recognition

This repository stores the source code for the hand gesture recognition model.

To use the model, download the following files: `requirements.txt`, `hand_gesture_reader.py`, `model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl`.
Next, enter the following command in the command prompt `pip install -r requirements.txt` to install the necessary libraries.

When the `hand_gesture_reader.py` file is run, a webcam window opens which can predict four gestures.
The gestures activate the following keys:
1. Hand Closed - up arrow key
2. Hand Three - right arrow key
3. Hand Open - left arrow key
4. Hand Zero - down arrow key

The sample images of gestures can be found at the end of README.

You may change the type of key to be activated by changing the `class_to_key` dictionary in the `hand_gesture_reader.py`. Just replace the dictionary values to a string representing the key you wish to activate. For example, you can change the key of the gesture `Closed` from `up` to `h` so that when the 'Hand Closed' gesture is shown, the program will activate the 'H' key. Refer to pyautogui documentation for the available keys.

The `hand_gesture_reader.py` file uses the random forest model parameters from `model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl` file. These parameters were trained in `model_hand_rf.py` file using the data stored in .npz files.

The data was made using `hand_landmark_dataset_maker.py` file.


Hand Zero
![hand_zero_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/ad40f4fc-0d5c-42c2-af1c-96e83f17b630)


Hand Closed
![hand_closed_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/8c2401aa-a8a5-4ed0-9240-ae662802468c)


Hand Open
![hand_open_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/eb28fa08-d13a-4ec6-bfbf-7f38b380cbf3)


Hand Three
![hand_three_sample_image](https://github.com/odil-T/Hand-Gesture-Recognition/assets/142138394/f8c36198-6003-4794-9e01-bf4bfe40f43d)
