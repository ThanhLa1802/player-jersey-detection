# import torchvision
# import torch
# print(torch.cuda.is_available())
# print(torchvision.ops.nms)
# print(torch.__version__)

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"runs\detect\train4\weights\best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict(r"video_test\Match_1824_1_0_subclip_3.mp4", save=True, imgsz=640, conf=0.65)