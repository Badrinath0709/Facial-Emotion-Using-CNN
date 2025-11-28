import os
import cv2
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
csv_path = r"C:\Users\badri\Downloads\archive (1)\emotions.csv"
images_dir = r"C:\Users\badri\Downloads\archive (1)\images"  # Folder with expression images
output_h5_path = r"C:\Users\badri\Downloads\archive (1)\emotions_dataset.h5"

# --- STEP 1: Load CSV ---
df = pd.read_csv(csv_path)
print("✅ CSV Loaded:", df.shape)
print(df.head())

# --- STEP 2: Prepare labels ---
# Use 'gender' or 'country' or 'emotion' if exists
if 'gender' in df.columns:
    label_column = 'gender'
elif 'emotion' in df.columns:
    label_column = 'emotion'
else:
    label_column = 'country'

le = LabelEncoder()
y = le.fit_transform(df[label_column])
print(f"✅ Encoded labels: {list(le.classes_)}")

# --- STEP 3: Load images ---
X = []
missing = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    img_filename = f"{row['set_id']}.jpg"  # assuming images are named like 0.jpg, 1.jpg etc.
    img_path = os.path.join(images_dir, img_filename)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (48, 48))  # resize to 48x48 pixels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
    else:
        missing.append(img_filename)

X = np.array(X, dtype="float32") / 255.0
y = np.array(y[:len(X)])

print(f"✅ Loaded {len(X)} images successfully. Missing: {len(missing)}")

# --- STEP 4: Save to HDF5 ---
with h5py.File(output_h5_path, "w") as h5f:
    h5f.create_dataset("X", data=X)
    h5f.create_dataset("y", data=y)
    h5f.create_dataset("label_classes", data=np.string_(le.classes_))

print(f"✅ Dataset saved to: {output_h5_path}")
print("Keys in file:", list(h5py.File(output_h5_path, 'r').keys()))
