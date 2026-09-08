import cv2
import numpy as np
import onnxruntime as ort
import winsound
from collections import deque
import time
from datetime import datetime
from cloud_uploader import cloud_send_async  # ✅ Added for cloud integration

# ================== SETTINGS ==================
MODEL_PATH = "yolov8n.onnx"
LABEL_PATH = "coco.names"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Focal length calibration (adjust after testing)
KNOWN_DISTANCE = 1.0  # meter
KNOWN_WIDTH = 0.5     # width of known object (e.g., shoulder width or A4 = 0.21m)
FOCAL_LENGTH = 615    # default, will be auto-updated if calibrated
# =================================================

# Load class labels
try:
    with open(LABEL_PATH, "r") as f:
        class_labels = [c.strip() for c in f.readlines() if c.strip()]
except FileNotFoundError:
    class_labels = [f"class_{i}" for i in range(80)]

# Load YOLOv8 ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Distance estimation
def estimate_distance(pixel_width):
    if pixel_width <= 0:
        return 999.0
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

# Beep alert safely (Windows)
def beep_alert(freq=1000, dur=300):
    try:
        winsound.Beep(freq, dur)
    except:
        pass

# Keep track of previous red alert state
red_alert_active = False
distance_smoothing = deque(maxlen=5)

# FPS helper
prev_time = time.time()

# ✅ Cloud update timer
last_cloud_send = 0.0
CLOUD_SEND_INTERVAL = 5.0   # seconds between cloud sends

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("✅ Running object detection with accurate distance estimation...")

def unify_nms_indices(idxs):
    """Return list of ints from cv2.dnn.NMSBoxes variety of returns."""
    if idxs is None:
        return []
    if isinstance(idxs, (list, tuple)):
        if len(idxs) == 0:
            return []
        if isinstance(idxs[0], (list, tuple, np.ndarray)):
            return [int(i[0]) for i in idxs]
        else:
            return [int(i) for i in idxs]
    if isinstance(idxs, np.ndarray):
        try:
            return idxs.flatten().tolist()
        except:
            return [int(x) for x in idxs]
    try:
        return [int(idxs)]
    except:
        return []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # compute FPS
    now = time.time()
    fps = 1.0 / (now - prev_time + 1e-8)
    prev_time = now

    h, w = frame.shape[:2]
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

    outputs = session.run(None, {input_name: img_input})
    output = outputs[0]

    # Handle YOLOv8 ONNX output shape
    if output.shape[1] == 84:
        output = np.transpose(output, (0, 2, 1))
    predictions = output[0]

    boxes, confidences, class_ids = [], [], []

    for pred in predictions:
        scores = pred[4:]
        cls_id = np.argmax(scores)
        conf = scores[cls_id]
        if conf > CONF_THRESHOLD:
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2) * w / 640)
            y1 = int((cy - bh / 2) * h / 640)
            bw = int(bw * w / 640)
            bh = int(bh * h / 640)
            boxes.append([x1, y1, bw, bh])
            confidences.append(float(conf))
            class_ids.append(int(cls_id))

    indices_raw = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
    indices = unify_nms_indices(indices_raw)

    current_red = False
    distances_in_frame = []

    for i in indices:
        if i < 0 or i >= len(boxes):
            continue
        x, y, bw, bh = boxes[i]
        cls_id = class_ids[i] if i < len(class_ids) else -1
        conf = confidences[i] if i < len(confidences) else 0.0

        distance = estimate_distance(bw)
        distance_smoothing.append(distance)
        distance_smoothed = float(np.mean(distance_smoothing))
        distances_in_frame.append(distance_smoothed)

        # Alert zones
        if distance_smoothed < 1.0:
            color = (0, 0, 255)
            alert_text = f"🚨 DANGER: {distance_smoothed:.2f} m"
            current_red = True
        elif 1.0 <= distance_smoothed <= 3.0:
            color = (0, 255, 255)
            alert_text = f"⚠️ WARNING: {distance_smoothed:.2f} m"
        else:
            color = (0, 255, 0)
            alert_text = f"✅ SAFE: {distance_smoothed:.2f} m"

        # Draw bounding box and labels
        label = f"{class_labels[cls_id] if (0 <= cls_id < len(class_labels)) else 'object'} {conf:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, alert_text, (x, y + bh + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Play beep only when entering red zone
    if current_red and not red_alert_active:
        beep_alert(1000, 400)
        red_alert_active = True
    elif not current_red:
        red_alert_active = False

    # --- HUD / Titles / Measurements ---
    panel_x, panel_y = 8, 8
    panel_w, panel_h = 360, 110
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 8, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,128), 2)
    det_count = len(indices)
    cv2.putText(frame, f"Detections: {det_count}", (panel_x + 8, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    if distances_in_frame:
        nearest = min(distances_in_frame)
        cv2.putText(frame, f"Nearest: {nearest:.2f} m", (panel_x + 8, panel_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    else:
        cv2.putText(frame, f"Nearest: N/A", (panel_x + 8, panel_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    title_text = "AI Pedestrian Detection - Low Light"
    title_x = max(10, (w // 2) - (len(title_text) * 6))
    cv2.putText(frame, title_text, (title_x, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, "Mode: Distance & Alert", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (w - 110, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # ✅ --- Cloud upload (non-blocking, throttled) ---
    try:
        if distances_in_frame:
            nearest_val = float(min(distances_in_frame))
            if nearest_val < 1.0:
                alert_str = "red"
            elif nearest_val <= 3.0:
                alert_str = "yellow"
            else:
                alert_str = "green"
        else:
            nearest_val = 999.0
            alert_str = "green"

        now_t = time.time()
        if now_t - last_cloud_send >= CLOUD_SEND_INTERVAL:
            det_count = len(indices)
            cloud_send_async(nearest_val, alert_str, det_count, fps)
            last_cloud_send = now_t
    except Exception as e:
        print("Cloud send error:", e)

    cv2.imshow("🔴 YOLOv8 Object Distance & Alerts Pro", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
