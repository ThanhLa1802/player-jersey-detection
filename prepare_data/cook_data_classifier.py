import os
import json
import cv2
from tqdm import tqdm
import shutil

# đường dẫn
json_root = "football_train"
image_root = "DATA_IMAGES"
output_root = "dataset_cls"

# clear output cũ
if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root, exist_ok=True)

# duyệt qua từng file JSON
for match_dir in os.listdir(json_root):
    match_path = os.path.join(json_root, match_dir)
    if not os.path.isdir(match_path):
        continue
    
    # tìm file json
    for f in os.listdir(match_path):
        if f.endswith(".json"):
            json_file = os.path.join(match_path, f)
            break
    else:
        continue
    
    # load annotation
    with open(json_file, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    # mapping image_id -> file_name
    image_map = {img["id"]: img["file_name"] for img in data["images"]}

    # crop từng annotation
    for ann in tqdm(data["annotations"], desc=match_dir):
        if ann["category_id"] != 4:  # chỉ lấy player
            continue
        attrs = ann.get("attributes", {})
        jersey_number = attrs.get("jersey_number", None)
        if jersey_number is None or jersey_number == "":
            continue
        
        # bbox [x,y,w,h]
        x, y, w, h = map(int, ann["bbox"])
        image_id = ann["image_id"]
        frame_name = image_map[image_id]
        
        # load ảnh
        img_path = os.path.join(image_root, match_dir, frame_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        # crop
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # lưu vào folder số áo
        out_dir = os.path.join(output_root, str(jersey_number))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{match_dir}_{image_id}_{ann['id']}.jpg")
        cv2.imwrite(out_path, crop)
