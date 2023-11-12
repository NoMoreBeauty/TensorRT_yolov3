from ultralytics import YOLO
import time
# Load a model
model = YOLO('./yolov3u.pt') 

model.predict('./samples/python/yolov3_onnx/dog.jpg', save=True)
