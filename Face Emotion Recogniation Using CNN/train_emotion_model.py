import pandas as pd
import numpy as np
import cv2
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ Load HDF5 file using pandas
h5_path = r"E:\t-hub\coding\project\Assessments\30-10-2025(Mini-Project)\emotion dataset\images\emotion_images.h5"
df = pd.read_hdf(h5_path, key='data')  # <-- use 'data'

print("✅ Loaded HDF5 dataset successfully!")
print(df.head())

# ----------------------------
# Load and preprocess images
# ----------------------------
images, labels = [], []

for _, row in df.iterrows():
    img_path = row['image_path']
    label = row['emotion']
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        labels.append(label)
    else:
        print(f"⚠️ Skipped unreadable image: {img_path}")

images = np.array(images).reshape(-1, 48, 48, 1) / 255.0
labels = np.array(labels)

# ----------------------------
# Encode labels
# ----------------------------
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# ----------------------------
# Split data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
)

print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ----------------------------
# Build CNN Model
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------
# Train Model
# ----------------------------
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ----------------------------
# Save Model
# ----------------------------
model.save("emotion_cnn_model.h5")
print("✅ Model trained and saved successfully as emotion_cnn_model.h5")


# watermark: Emotion Model Trainer
print("📝 Script by Emotion Model Trainer")