import json
import os

def convert_annotations_to_yolo(json_path, output_dir, target_classes=[3, 4]):
    """
    Chuyển đổi annotation từ định dạng COCO sang YOLO.

    Args:
        json_path (str): Đường dẫn tới file annotation (.json).
        output_dir (str): Thư mục để lưu các file .txt của YOLO.
        target_classes (list): Danh sách category_id muốn giữ lại (VD: 3: ball, 4: player).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]

        if category_id not in target_classes:
            continue

        image = images[image_id]
        img_w, img_h = image["width"], image["height"]
        file_name = image["file_name"].replace(".PNG", ".txt")
        label_path = os.path.join(output_dir, file_name)

        x, y, w, h = ann["bbox"]
        x_center = x + w / 2
        y_center = y + h / 2

        # Normal hóa
        x_center /= img_w
        y_center /= img_h
        w /= img_w
        h /= img_h

        # Điều chỉnh class_id: 3 → 0, 4 → 1 (theo YOLO nếu cần)
        if category_id == 3:  # ball
            class_id = 0
        elif category_id == 4:  # player
            class_id = 1
        else:
            continue

        with open(label_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ Done converting annotations to YOLO format in: {output_dir}")

# Example usage
# create a folder to save the labels for all videos
# if not os.path.exists("labels"):
#     os.makedirs("labels")

# json_path = "./football_train/Match_1824_1_0_subclip_3/Match_1824_1_0_subclip_3.json"
# output_dir = "labels"
# convert_annotations_to_yolo(json_path, output_dir)

if __name__ == "__main__":
    # Convert annotations for all videos in a folder
    folder_path = "./football_train"
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            for json_file in os.listdir(sub_folder_path):
                if json_file.endswith(".json"):
                    json_path = os.path.join(sub_folder_path, json_file)
                    output_dir = os.path.join("labels", sub_folder)
                    convert_annotations_to_yolo(json_path, output_dir)