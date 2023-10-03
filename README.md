# Hand-Gesture-Recognition

This repository stores the source code for the hand gesture recognition model.

Please note that you will need python and pip to be installed.
To use the model, download the following files: `requirements.txt`, `hand_gesture_reader.py`, `model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl`.
Next, enter the following command in the command prompt `pip install -r requirements.txt` to install the necessary libraries.

When the `hand_gesture_reader.py` file is run, a webcam window opens which can predict four gestures.
The gestures activate the following keys:
1. Hand Closed - up arrow key
2. Hand Three - right arrow key
3. Hand Open - left arrow key
4. Hand Zero - down arrow key

The sample images of gestures can be found in sample_images folder.

You may change the type of key to be activated by changing the `class_to_key` dictionary in `hand_gesture_reader.py` file. Just replace the dictionary values to a string representing the key you wish to activate. For example, you can change the key of the gesture `Closed` from `up` to `h` so that when the 'Hand Closed' gesture is shown, the program will activate the 'H' key. Refer to pyautogui documentation for the available keys.

The `hand_gesture_reader.py` file uses the random forest model parameters from `model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl` file. These parameters were trained in `model_hand_rf.py` file using the data stored in .npz files.

The data was made using `hand_landmark_dataset_maker.py` file.




Hand Closed
<img src='sample_images/Hand Closed.jpg'/>


Hand Three
<img src='sample_images/Hand Three.jpg'/>


Hand Open
<img src='sample_images/Hand Open.jpg'/>


Hand Zero
<img src='sample_images/Hand Zero.jpg'/>
