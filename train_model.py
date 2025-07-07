import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
df = pd.read_csv("combined_landmarks.csv")
X = df.drop('label', axis=1).values
y = df['label'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training model...")
model.fit(X_train, y_train, epochs=26, batch_size=32, validation_data=(X_test, y_test))
model.save("sign_language_model.h5")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model and label encoder saved.")