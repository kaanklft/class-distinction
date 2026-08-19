import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Mevcut çalışma dizinini tam yol olarak alıyoruz
    dataset_path = os.path.abspath('.')

    # YOLOv8 Sınıflandırma modelini yüklüyoruz
    model = YOLO('yolov8n-cls.pt')

    # Eğitimi başlatıyoruz
    results = model.train(
        data=dataset_path,  # Tam dosya yolunu vererek dizin karışıklığını önlüyoruz
        epochs=50, #epochs nedir = 50, # tur sayısı
        imgsz=224, #imgsz = 224, # Görüntü boyutu 224x224 olarak ayarlanıyor
        batch=32, #batch nedir = 32, paketleme boyutu 32
        workers=4, #workers nedir = 4, # 
        device=0            # GPU hazır olduğu için 0 olarak kalabilir
    )