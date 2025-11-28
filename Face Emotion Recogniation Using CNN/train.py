import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
with h5py.File(r"E:\t-hub\coding\project\Assessments\30-10-2025(Mini-Project)\emotion dataset\images\emotion_dataset.h5", "r") as h5f:
    X = np.array(h5f["images"])
    y = np.array(h5f["labels"])

print(f"Dataset loaded: X={X.shape}, y={y.shape}")

# Normalize pixel values
X = X / 255.0

# Encode labels
# le = LabelEncoder()
#y = le.fit_transform(y)  # Convert text to integer (e.g., Happy -> 0, Sad -> 1)
#y = to_categorical(y)    # Convert integers to one-hot vectors

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure correct shape for grayscale images
if X_train.ndim == 3:
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

# Build CNN model
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("\n🚀 Training started...\n")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save trained model
model.save("emotion_cnn_model.h5")
print("\n✅ Training complete. Model saved as emotion_cnn_model.h5")
