import os
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime

# Вывод метрик на экран после обучения
def find_latest_directory(directory):
    items = os.listdir(directory)
    directories = [os.path.join(directory, item) for item in items if os.path.isdir(os.path.join(directory, item))]
    if not directories:
        return None
    latest_directory = max(directories, key=os.path.getctime)
    
    return latest_directory

def display_image(image_path):
    window = tk.Tk()
    window.title("Отображение изображения")

    img = Image.open(image_path)
    
    window_width = 800
    window_height = 400
    window.geometry(f"{window_width}x{window_height}")

    img = img.resize((window_width, window_height), Image.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(img)

    label = tk.Label(window, image=img_tk)
    label.pack(expand=True)

    window.mainloop()

model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

if __name__ ==  '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results = model.train(data=os.path.join(script_dir, 'dataset_teeth6', 'data.yaml'), epochs=3, imgsz=640, workers = 2, lr0=0.001)
    runs_dir = os.path.join(script_dir, "runs", "detect")
    image_path = os.path.join(find_latest_directory(runs_dir), "results.png")
    display_image(image_path)
