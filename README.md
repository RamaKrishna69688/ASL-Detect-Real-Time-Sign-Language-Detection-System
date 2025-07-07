# ASL Detect: Real-Time Sign Language Detection System

## Project Overview

**ASL Detect** is a real-time American Sign Language (ASL) detection system that recognizes hand signs from live webcam input and converts them into text sentences. This project leverages MediaPipe for hand landmark detection and a deep learning model trained on ASL alphabets and additional gestures like **"delete"**, **"space"**, and **"nothing"** to provide accurate, continuous sign recognition.

---

## Features

* **Data Collection**: Extracts 3D hand landmark features from labeled images across multiple ASL gesture classes.
* **Model Training**: Trains a neural network to classify hand landmarks into ASL letters and commands.
* **Real-Time Detection**: Detects hand signs from webcam video and builds readable sentences using gesture and pause logic.

---

## Technologies Used

* Python 3.x
* OpenCV
* MediaPipe Hands
* TensorFlow / Keras
* Scikit-learn
* NumPy, Pandas
* Pickle (for saving label encoder)

---

## Project Structure

1. **`data_collection.py`**
   Processes images from labeled folders and saves 3D hand landmarks into a CSV file.

2. **`train_model.py`**
   Loads landmark data from CSV, trains a neural network model, and saves the model and label encoder.

3. **`real_time_detection.py`**
   Uses webcam input and the trained model to detect ASL signs in real time and construct readable sentences.

---

## How to Use

### Step 1: Prepare Dataset

* Download the ASL Alphabet dataset (e.g., from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)).
* After downloading the ZIP file, **extract it into a normal folder** (e.g., `asl_alphabet_train`).
* Ensure the folder contains subfolders for each label: `A-Z`, `del`, `space`, and `nothing`.
* Run `data_collection.py` to generate a CSV file with hand landmark data (`combined_landmarks.csv`).

### Step 2: Train the Model

* Run `train_model.py`.
* This will train the model using the extracted landmark data and save:

  * `sign_language_model.h5` (the trained model)
  * `label_encoder.pkl` (the label-to-class converter)

### Step 3: Run Real-Time Detection

* Run `real_time_detection.py`.
* Show ASL signs in front of your webcam.
* The system will detect signs, build a sentence, and display it on the screen.

---

## Notes

* Use clear and consistent hand gestures in the dataset.
* The model expects 21 hand landmarks, each with x, y, z coordinates (total 63 features).
* If no hand is detected for a short time, the system automatically inserts a space into the sentence.

---

## Future Improvements

* Add support for backspace or correction gestures.
* Integrate voice output (text-to-speech) for detected sentences.
* Extend detection to support dynamic gestures or two-handed signs.
* Build a simple graphical interface for improved usability.

---

## Acknowledgments

* Dataset Source: [ASL Alphabet on Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)
* [MediaPipe](https://google.github.io/mediapipe/) by Google for real-time hand tracking
* [TensorFlow/Keras](https://www.tensorflow.org/) for training the deep learning model

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it for personal or academic use.

---
