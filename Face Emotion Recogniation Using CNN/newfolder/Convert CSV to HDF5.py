import pandas as pd
import os

# ✅ Correct path (note the double backslash or raw string)
csv_path = r"D:\Badri\New folder (2)\Facial-Emotion-Using-CNN\Face Emotion Recogniation Using CNN\emotion dataset\images\emotion_images.csv"

# ✅ Check if file exists before reading
if not os.path.exists(csv_path):
    print(f"❌ CSV file not found at: {csv_path}")
else:
    df = pd.read_csv(csv_path)

    # ✅ Convert to HDF5 (.h5)
    h5_path = csv_path.replace('.csv', '.h5')
    df.to_hdf(h5_path, key='data', mode='w')

    print(f"✅ H5 file created successfully: {h5_path}")
