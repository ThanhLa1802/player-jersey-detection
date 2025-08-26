# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO("yolo11n.pt")

# # Run inference on 'bus.jpg' with arguments
# model.predict("football_train\Match_1824_1_0_subclip_3\Match_1824_1_0_subclip_3.mp4", save=True, imgsz=640, conf=0.3, iou=0.6)

from ultralytics import YOLO

# Load a model

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(
    data="yolo_dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=128,
    patience=20,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    device=0,
    plots=True
    )