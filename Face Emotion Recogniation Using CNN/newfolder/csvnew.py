import os
import pandas as pd

# Path to your main folder that contains emotion folders
base_dir = r"E:\t-hub\coding\project\Assessments\30-10-2025(Mini-Project)\emotion dataset\images"  # Change if needed

data = []

# Loop through all subfolders (emotions)
for emotion_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, emotion_folder)
    if os.path.isdir(folder_path):  # Ensure it's a folder
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                data.append([emotion_folder, img_path])

# Create CSV file
df = pd.DataFrame(data, columns=['emotion', 'image_path'])
csv_path = os.path.join(base_dir, "emotion_images.csv")
df.to_csv(csv_path, index=False)

print(f"✅ CSV file created successfully: {csv_path}")
print(df.head())
