import cv2
import csv
import os
import mediapipe as mp

data_dir = /*Mention the location fo the file*/
csv_filename = "combined_landmarks.csv"
labels = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ["del", "space", "nothing"]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
    csv_writer.writerow(header)
    for label in labels:
        print(f"\n📂 Processing: {label}")
        label_folder = os.path.join(data_dir, label)
        if not os.path.exists(label_folder):
            print(f"⚠️ Folder for '{label}' not found. Skipping.")
            continue
        image_files = [f for f in os.listdir(label_folder) if f.lower().endswith(('.jpg', '.png'))]
        if not image_files:
            print(f"⚠️ No images found for '{label}'. Skipping.")
            continue
        for img_name in image_files:
            img_path = os.path.join(label_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(label)
                csv_writer.writerow(row)
        print(f"✅ Processed {len(image_files)} images for '{label}'")
hands.close()
print(f"\n🎉 All done! Landmark data saved to: {csv_filename}")