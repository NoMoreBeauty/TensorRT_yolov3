from ultralytics import YOLO
import time
# Load a model
model = YOLO('./yolov3u.pt') 
timestampstart = time.time()
model.predict('./samples/python/yolov3_onnx/dog.jpg', save=True)
timestampend = time.time()
print(timestampend-timestampstart)