import os
import shutil
import random

# Thư mục chứa dữ liệu gốc
images_root = "extracted_frames"
labels_root = "labels"

# Dataset đầu ra
out_dir = "dataset"
for split in ["train", "val"]:
    os.makedirs(f"{out_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{out_dir}/labels/{split}", exist_ok=True)

# Gom tất cả ảnh và nhãn vào danh sách
all_images = []
for match_folder in os.listdir(images_root):
    match_path = os.path.join(images_root, match_folder)
    if not os.path.isdir(match_path):
        continue
    for img_file in os.listdir(match_path):
        if img_file.endswith(".PNG"):
            img_path = os.path.join(match_path, img_file)
            label_path = os.path.join(labels_root, match_folder, img_file.replace(".PNG", ".txt"))
            if os.path.exists(label_path):
                all_images.append((img_path, label_path))

print("Tổng số ảnh:", len(all_images))

# Chia 80% train, 20% val
random.shuffle(all_images)
train_size = int(0.8 * len(all_images))
train_data = all_images[:train_size]
val_data = all_images[train_size:]

def copy_data(data, split):
    for img_path, label_path in data:
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        shutil.copy(img_path, f"{out_dir}/images/{split}/{img_name}")
        shutil.copy(label_path, f"{out_dir}/labels/{split}/{label_name}")

copy_data(train_data, "train")
copy_data(val_data, "val")

print("Hoàn tất dataset YOLO!")
