import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from PIL import Image
import torch
import cv2
import numpy as np

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # Загрузка модели YOLO из папки относительно пути скрипта
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, "last.pt")  
        self.model = YOLO(model_path)

        self.class_colors = {
            0: (255, 0, 0),  # Красный для класса 0
            1: (0, 255, 0),  # Зелёный для класса 1
            2: (0, 0, 255),  # Синий для класса 2
            3: (255, 255, 0),  # Жёлтый для класса 3
        }

    def initUI(self):
        self.setWindowTitle('Teeth Anomalies Detection')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel(self)
        self.label.setText('Загрузите изображение для предсказания')
        self.label.setAlignment(Qt.AlignCenter)  # Выровнять текст по центру
        layout.addWidget(self.label)

        self.button = QPushButton('Загрузить изображение', self)
        self.button.clicked.connect(self.loadImage)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def loadImage(self):
        # Открытие диалогового окна для выбора изображения
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpeg *.jpg *.webp);;All Files (*)", options=options)
        
        if file_path:
            # Отображение изображения в интерфейсе
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height()))
            self.predict(file_path)

    def predict(self, image_path):
        # Используем модель YOLO для предсказания
        results = self.model(image_path)

        # Извлечение боксов и обработка изображения с предсказаниями
        img = cv2.imread(image_path)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Координаты боксов
                confidence = box.conf.item()  # Уверенность предсказания
                cls = int(box.cls.item())  # Класс объекта
                if cls == 0: # Если класс Healthy, тогда не отрисовываем
                    break

                # Получаем цвет, соответствующий классу
                color = self.class_colors.get(cls, (255, 255, 255))

                # Отрисовка боксов на изображении
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{self.model.names[cls]}: {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Отображение изображения с боксами
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save("output_with_boxes.jpg")
        pixmap_with_boxes = QPixmap("output_with_boxes.jpg")
        self.label.setPixmap(pixmap_with_boxes.scaled(self.label.width(), self.label.height()))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YOLOApp()
    ex.show()
    sys.exit(app.exec_())
