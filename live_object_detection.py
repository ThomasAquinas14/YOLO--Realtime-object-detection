"""Live object detection from a webcam using a pre-trained YOLO11 model.

Run with:  python live_object_detection.py
Press 'q' to quit.
"""

from ultralytics import YOLO
import cv2
import torch

# start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0). Is a camera connected and free?")
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolo11s.pt")

# only show detections at or above this confidence (raise to suppress bad guesses)
CONF_THRESHOLD = 0.5

# run on GPU if available, otherwise CPU
device = 0 if torch.cuda.is_available() else "cpu"

# object classes (COCO labels, provided by the model)
classNames = model.names

try:
    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, device=device, conf=CONF_THRESHOLD, verbose=False)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # class name and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{classNames[cls]} {conf:.2f}"

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, label, org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
