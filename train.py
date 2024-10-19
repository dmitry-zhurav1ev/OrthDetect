import os
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

if __name__ ==  '__main__':
    results = model.train(data=os.path.join(script_dir, 'dataset_teeth', 'data.yaml'), epochs=50, imgsz=640, workers = 2, lr0=0.001)