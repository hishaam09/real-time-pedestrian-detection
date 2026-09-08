# calibrate_focal.py
import cv2, json, time, os
from ultralytics import YOLO
import numpy as np

MODEL = "optimized_models/yolov8n.onnx"  # path to your ONNX
OUT_JSON = "focal_data.json"
IMG_SIZE = 640
CONF_THRESH = 0.5
DEFAULT_CLASS = "person"

# expected real heights (meters) - you can edit/add classes
KNOWN_HEIGHTS = {
    "person": 1.7,
    "car": 1.5,
    "bicycle": 1.2
}

model = YOLO(MODEL)
cap = cv2.VideoCapture(0)

samples = {}  # class -> list of focal estimates

print("Calibration tool\nPress 'c' to capture current largest detection bounding box.\nPress 'q' to quit and save averages.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    results = model.predict(img, imgsz=IMG_SIZE, conf=CONF_THRESH)
    # draw boxes on preview (uses ultralytics result format)
    display = frame.copy()
    max_box = None
    max_area = 0
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)
            if area > max_area:
                max_area = area
                max_box = (x1,y1,x2,y2, int(box.cls[0]), float(box.conf[0]))
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 1)
    if max_box:
        x1,y1,x2,y2,cls,conf = max_box
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(display, f"Class:{cls} Conf:{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Calibration - press c to capture", display)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        if not max_box:
            print("No detection to capture — try again.")
            continue
        px_h = (max_box[3] - max_box[1])  # pixel height on original frame
        cls_id = max_box[4]
        cls_name = model.names[cls_id] if cls_id in model.names else DEFAULT_CLASS
        print(f"Captured {cls_name} bbox height (px): {px_h}")
        # ask user for actual distance in meters
        d = input("Enter measured distance to object (meters), e.g. 2.0: ").strip()
        try:
            d = float(d)
        except:
            print("Invalid distance, skipping sample.")
            continue
        # known real height for this class (if available)
        real_h = KNOWN_HEIGHTS.get(cls_name, None)
        if real_h is None:
            ans = input(f"No known height for class '{cls_name}'. Enter real height in meters (or blank to skip): ").strip()
            if not ans:
                continue
            real_h = float(ans)
            KNOWN_HEIGHTS[cls_name] = real_h

        # focal estimate (pixels)
        f_est = (px_h * d) / real_h
        print(f"Estimated focal (px) for this sample: {f_est:.2f}")
        samples.setdefault(cls_name, []).append(f_est)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# average and save
focal_data = {}
for cls, vals in samples.items():
    avg = sum(vals)/len(vals)
    focal_data[cls] = {"focal": avg, "samples": vals}

with open(OUT_JSON, "w") as f:
    json.dump({"focal_by_class": focal_data, "KNOWN_HEIGHTS": KNOWN_HEIGHTS}, f, indent=2)

print(f"Saved calibration into {OUT_JSON}")
print("Contents:", focal_data)
