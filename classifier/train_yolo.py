
from ultralytics import YOLO

# Load a model

if __name__ == "__main__":
    model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
     # Train
    model.train(
        data="dataset_cls",
        epochs=30,
        imgsz=224,
        batch=128
    )