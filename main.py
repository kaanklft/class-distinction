import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Get the current working directory as an absolute path
    dataset_path = os.path.abspath('.')

    # Load the YOLOv8 classification model
    model = YOLO('yolov8n-cls.pt')

    # Start training
    results = model.train(
        data=dataset_path,  # Pass the absolute path to avoid directory confusion
        epochs=50,          # number of training epochs
        imgsz=224,          # images are resized to 224x224
        batch=32,            # batch size
        workers=4,           # number of dataloader worker threads
        device=0              # GPU is available, so this stays 0
    )
