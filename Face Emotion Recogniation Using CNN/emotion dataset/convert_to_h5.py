import os
import cv2
import numpy as np
import h5py
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ===============================
# 🔧 PATH CONFIGURATION
# ===============================
BASE_PATH = r"D:\Badri\New folder (2)\Facial-Emotion-Using-CNN\Face Emotion Recogniation Using CNN\emotion dataset\images"   # Path to dataset with emotion folders
OUTPUT_H5 = os.path.join(BASE_PATH, "emotion_dataset.h5")
IMAGE_SIZE = (48, 48)

# ===============================
# 📸 LOAD IMAGES FROM FOLDERS
# ===============================
print("🔹 Scanning emotion folders...")

images = []
labels = []

for emotion_folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, emotion_folder)

    # Only process subfolders (ignore files)
    if os.path.isdir(folder_path):
        print(f"📁 Loading images from: {emotion_folder}")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"⚠️ Skipping unreadable image: {filename}")
                    continue

                # Convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Resize all images to 48x48
                img = cv2.resize(img, IMAGE_SIZE)
                # Normalize pixel values
                img = img / 255.0

                images.append(img)
                labels.append(emotion_folder)

images = np.array(images, dtype=np.float32)
images = np.expand_dims(images, -1)  # Add channel dimension

# ===============================
# 🏷️ ENCODE LABELS
# ===============================
print("\n🔹 Encoding emotion labels...")
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# ===============================
# 💾 SAVE TO HDF5 FILE
# ===============================
print("\n💾 Saving dataset to HDF5 file...")
with h5py.File(OUTPUT_H5, "w") as h5f:
    h5f.create_dataset("images", data=images)
    h5f.create_dataset("labels", data=categorical_labels)
    h5f.create_dataset("label_names", data=np.bytes_(le.classes_))

print("\n🎉 Saved successfully!")
print(f"📂 File: {OUTPUT_H5}")
print(f"🧠 Emotions: {list(le.classes_)}")
print(f"📸 Total images: {images.shape[0]}")

