import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

# YOLOv8 modelini indir ve yükle
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# Canlı kamera akışı için OpenCV'yi başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera içindir. Harici kameralar için 1, 2 gibi başka indeksler kullanılabilir.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera akışı alınamadı.")
        break

    # YOLOv8 ile tahmin yap
    results = model(frame)  # OpenCV'nin verdiği frame doğrudan kullanılabilir
    detections = Detections.from_ultralytics(results[0])

    # Tespit edilen nesneleri işaretle
    for box in detections.xyxy:  # x_min, y_min, x_max, y_max koordinatlarını içerir
        x_min, y_min, x_max, y_max = map(int, box[:4])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Yeşil dikdörtgen çiz

    # Görüntüyü OpenCV penceresinde göster
    cv2.imshow("YOLOv8 Face Detection", frame)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()