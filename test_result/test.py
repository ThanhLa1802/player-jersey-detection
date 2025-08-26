import cv2
from ultralytics import YOLO

# Load models
det_model = YOLO("best_detect.pt")    # model detection cầu thủ (category: player)
cls_model = YOLO("best_cls.pt")       # model classification số áo

# Input & output video
video_path = "input.mp4"
output_path = "output_with_jersey.mp4"

cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect players
    results = det_model(frame, conf=0.5)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Classification số áo
            cls_results = cls_model.predict(crop, imgsz=224)
            jersey = cls_results[0].probs.top1  # class id
            conf = cls_results[0].probs.top1conf.item()

            # Vẽ bounding box + số áo
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            
            label = f"{jersey}"

            # Viết text viền đen trước (dày hơn)
            cv2.putText(frame, label, (x1, y1-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 6)

            # Viết text màu vàng chồng lên
            cv2.putText(frame, label, (x1, y1-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 4)

    out.write(frame)

cap.release()
out.release()
print("✅ Done! Video saved at:", output_path)
