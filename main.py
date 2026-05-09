import os
import shutil

# Path to your main folder containing all images
source_dir = r"C:\Users\Khushan\Desktop\segment\archive\train"

# Create destination folders
sat_dir = os.path.join(source_dir, "sat")
mask_dir = os.path.join(source_dir, "mask")

os.makedirs(sat_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Loop through files
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    # Skip folders
    if os.path.isdir(file_path):
        continue

    # Move based on filename
    if "_sat" in filename.lower():
        shutil.move(file_path, os.path.join(sat_dir, filename))

    elif "_mask" in filename.lower():
        shutil.move(file_path, os.path.join(mask_dir, filename))

print("✅ Files successfully organized into 'sat' and 'mask' folders.")
