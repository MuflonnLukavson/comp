import cv2
import time
import numpy as np
from torch import classes
import json

cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://10.100.33.246:4747/video") ## Change the IP for connecting the phone camera
font = cv2.FONT_HERSHEY_SIMPLEX

if not cap.isOpened():
    raise RuntimeError("Could not open the camera or stream")


t1 = 0
t2 = 0
ret, frame = cap.read()


H, W, C = frame.shape
fps = 30
count = 0

if hasattr(cv2.dnn, "ENGINE_OPENCV"):
    DNN_ENGINE = cv2.dnn.ENGINE_OPENCV
elif hasattr(cv2.dnn, "ENGINE_NEW"):
    DNN_ENGINE = cv2.dnn.ENGINE_NEW
else:
    raise RuntimeError("This example requires OpenCV 5's built-in DNN engine.")

PATH = "/home/lucky/ru_erasmus/comp/yolo26/yolo26n.onnx"
model = cv2.dnn.readNetFromONNX(PATH, DNN_ENGINE)


with open("/home/lucky/ru_erasmus/comp/yolo26/coco80.json") as stream:
    classes = json.load(stream)

size = 640



while(True):
    start_time = time.time()
    ret, frame = cap.read()
    capture_time = time.time() - start_time
    count += 1

    # frame = cv2.resize(frame, (H // 2, W // 2))

    if count == 5:
        t1 = t2
        t2 = time.time() 
        fps = 5 / (t2 - t1)
        count = 0


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    display = frame.copy()

    gain = min(size / W, size / H)
    new_width, new_height = round(W * gain), round(H * gain)
    resized = cv2.resize(frame, (new_width, new_height))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    padded = cv2.copyMakeBorder(
    resized, top, size - new_height - top, left, size - new_width - left,
    cv2.BORDER_CONSTANT, value=(114, 114, 114),
)

    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True, crop=False)

    model.setInput(blob)
    output = model.forward()

    detections = output[0][output[0, :, 4] >= 0.35].copy()
    detections[:, [0, 2]] = (detections[:, [0, 2]] - left) / gain
    detections[:, [1, 3]] = (detections[:, [1, 3]] - top) / gain
    detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, W)
    detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, H)

    for x1, y1, x2, y2, score, class_id in detections:
        if x2 <= x1 or y2 <= y1:
            continue
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        label = f"{classes[int(class_id)]} {score:.2f}"
        cv2.rectangle(display, p1, p2, (40, 220, 40), 2)
        cv2.putText(display, label, (p1[0], max(p1[1] - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 40), 2, cv2.LINE_AA)

    cv2.putText(display, f'fps: {fps:.1f}', (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)

    cv2.imshow('frame',display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break