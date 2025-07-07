# ASL-Detect-Real-Time-Sign-Language-Detection-System

## Project Overview

ASL Detect is a real-time American Sign Language (ASL) detection system that recognizes hand signs from live webcam input and converts them into text sentences. This project leverages MediaPipe for hand landmark detection and a deep learning model trained on ASL alphabets and additional gestures like "delete", "space", and "nothing" to provide accurate, continuous sign recognition.

## Features

* Data Collection: Extracts 3D hand landmark features from images across multiple ASL gesture classes including alphabets and special signs.
* Model Training: Trains a neural network to classify hand landmarks into ASL letters and commands.
* Real-Time Detection: Detects hand signs from webcam video and builds readable sentences with pause and gesture handling.

## Technologies Used

* Python 3.x
* OpenCV
* MediaPipe Hands
* TensorFlow / Keras
* Scikit-learn
* NumPy, Pandas
* Pickle (for model label encoding)

## Project Structure

1. **Data Collection (`data_collection.py`)**: Processes images from labeled folders and saves hand landmarks to CSV.
2. **Model Training (`train_model.py`)**: Loads CSV data, trains a neural network, and saves the model and label encoder.
3. **Real-Time Detection (`real_time_detection.py`)**: Uses webcam feed and the trained model to detect signs and display sentences live.

## How to Use

### Step 1: Prepare Dataset

Organize your dataset folder with subfolders for each label (A-Z, del, space, nothing). Run `data_collection.py` to generate the CSV file.

### Step 2: Train Model

Run `train_model.py` to train the neural network. The model and label encoder files will be saved.

### Step 3: Run Real-Time Detection

Run `real_time_detection.py` to start webcam sign recognition and see sentences form in real time.

## Notes

* Ensure clear, varied hand poses in the dataset.
* The model expects 21 hand landmarks (x, y, z coordinates).
* Pauses in hand detection add spaces to the sentence.

## Future Improvements

* Support backspace gesture and sentence correction.
* Add voice output for detected sentences.
* Recognize two-handed or dynamic gestures.
* Develop a GUI for enhanced user experience.

## Acknowledgments

* ASL Alphabet dataset ([https://www.kaggle.com/grassknoted/asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet))
* MediaPipe for hand landmark detection.
* TensorFlow for deep learning tools.

## License

This project is licensed under the MIT License.
