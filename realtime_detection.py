import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time

model = tf.keras.models.load_model("sign_language_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
print("Starting real-time sign detection. Press 'q' to quit.")
sentence = ""
last_prediction = ""
confirmed_prediction = ""
waiting_for_hand_removal = False
no_hand_start_time = None
no_hand_duration_required = 1.0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        prediction = model.predict(np.array([landmarks]))[0]
        predicted_label = le.classes_[np.argmax(prediction)]
        if not waiting_for_hand_removal:
            if predicted_label != confirmed_prediction:
                sentence += predicted_label
                confirmed_prediction = predicted_label
                print(f"✅ Added: {predicted_label}")
                print(f"Current Sentence: {sentence}")
                waiting_for_hand_removal = True
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        no_hand_start_time = None
    else:
        if waiting_for_hand_removal:
            if no_hand_start_time is None:
                no_hand_start_time = time.time()
            elif time.time() - no_hand_start_time > no_hand_duration_required:
                waiting_for_hand_removal = False
                confirmed_prediction = ""
                print("✋ Hand removed, ready for next letter.")
    cv2.putText(frame, f'Sentence: {sentence}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Real-Time Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
hands.close()