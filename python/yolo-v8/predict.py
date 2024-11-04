from ultralytics import YOLO
model = YOLO("./yolov8s.safetensors")
results = model.predict(source="./", save=True)