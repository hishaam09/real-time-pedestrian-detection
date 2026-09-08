# advanced_distance_alert.py
# Robust, low-light, threaded YOLOv8 ONNX detection with distance (m) + alerts + calibration

import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import json
import os
import winsound
from collections import deque
from datetime import datetime
import sys

# ---------------- CONFIG ----------------
ONNX_MODEL = "yolov8n.onnx"    # must exist
COCO_NAMES = "coco.names"      # must exist (80 class names)
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4
NMS_IOU = 0.45

# Distance estimation using bbox width:
KNOWN_WIDTH = 0.5   # meters — average person shoulder width; adjust if needed
FOCAL_FILE = "focal_data.json"
FOCAL_LENGTH = 615.0  # fallback; will be overwritten by calibration if provided

# Alert thresholds (meters)
RED_THRESH = 1.0     # <= 1.0m -> RED
YELLOW_THRESH = 3.0  # >1 and <=3 -> YELLOW
# GREEN: >3.0m

SMOOTH_WINDOW = 5    # smoothing window for distances (per object)
CAM_ID = 0
# ----------------------------------------

# ---------- Threaded camera to reduce lag ----------
class ThreadedCam:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot open camera.")
        self.ret = ret
        self.frame = frame
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=0.5)
        except:
            pass
        self.cap.release()

# ---------- Low-light enhancement (CLAHE + gamma) ----------
def enhance_lowlight(frame, clip_limit=3.0, tile_grid=(8,8), gamma=1.15):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ---------- Load labels ----------
if os.path.exists(COCO_NAMES):
    with open(COCO_NAMES, "r") as f:
        CLASS_NAMES = [x.strip() for x in f.readlines() if x.strip()]
else:
    CLASS_NAMES = [f"class_{i}" for i in range(80)]
    print("Warning: coco.names not found — using placeholders.", file=sys.stderr)

# ---------- Load focal calibration if exists ----------
GLOBAL_FOCAL = FOCAL_LENGTH
focal_by_class = {}
known_heights = {}
if os.path.exists(FOCAL_FILE):
    try:
        with open(FOCAL_FILE, "r") as f:
            data = json.load(f)
            focal_by_class = data.get("focal_by_class", {})
            known_heights = data.get("KNOWN_HEIGHTS", {})
            # compute average focal
            fls = []
            for v in focal_by_class.values():
                if isinstance(v, dict) and "focal" in v:
                    fls.append(float(v["focal"]))
                else:
                    try:
                        fls.append(float(v))
                    except:
                        pass
            if fls:
                GLOBAL_FOCAL = float(np.mean(fls))
            print("Loaded calibration:", focal_by_class)
    except Exception as e:
        print("Could not load focal_data.json:", e)

# ---------- Load ONNX model ----------
if not os.path.exists(ONNX_MODEL):
    print("ERROR: ONNX model missing:", ONNX_MODEL, file=sys.stderr)
    sys.exit(1)

print("Loading ONNX model...")
session = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model loaded.")

# ---------- Robust parser for YOLOv8 ONNX output ----------
def parse_yolov8_output(output, conf_thr=CONF_THRESHOLD):
    """
    Accepts raw ONNX output and returns lists: boxes_norm (x_c,y_c,w,h), scores, class_ids
    Handles common shapes (1,84,8400) or (1,8400,85).
    """
    out = output
    if out.ndim == 3 and out.shape[1] in (84,85):
        out = np.transpose(out, (0,2,1))
    preds = out[0]  # (N, D)
    boxes, scores, cids = [], [], []
    D = preds.shape[1]
    for p in preds:
        if D >= 85:
            x,y,w,h,obj = p[:5]
            cls_scores = p[5:]
            cid = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cid])
            score = float(obj * cls_conf)
            if score < conf_thr:
                continue
            boxes.append([float(x), float(y), float(w), float(h)])
            scores.append(score)
            cids.append(cid)
        elif D == 84:
            x,y,w,h,obj = p[:5]
            cls_scores = p[5:]
            cid = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cid])
            score = float(obj * cls_conf)
            if score < conf_thr:
                continue
            boxes.append([float(x), float(y), float(w), float(h)])
            scores.append(score)
            cids.append(cid)
        else:
            # unknown format -> skip
            continue
    return boxes, scores, cids

# ---------- Utility: estimate distance using bbox width ----------
def estimate_distance_from_width(pixel_width, known_width_m=KNOWN_WIDTH, focal_px=GLOBAL_FOCAL):
    if pixel_width <= 0:
        return float('inf')
    # distance (m) = (known_width * focal) / pixel_width
    return (known_width_m * focal_px) / (pixel_width + 1e-6)

# ---------- smoothing helper keyed by coarse center grid ----------
distance_queues = {}  # key -> deque

def smooth_distance_for_bbox(cx, cy, dist):
    # key by rounded center to avoid ephemeral keys
    key = (int(cx//20), int(cy//20))
    dq = distance_queues.get(key, deque(maxlen=SMOOTH_WINDOW))
    dq.append(dist)
    distance_queues[key] = dq
    return float(np.mean(dq))

# ---------- beep helper ----------
def beep_once(freq=1200, duration=300):
    try:
        winsound.Beep(int(freq), int(duration))
    except Exception:
        pass

# ---------- NMS results normalizer ----------
def unify_nms_indices(idxs):
    if idxs is None:
        return []
    if isinstance(idxs, (list, tuple)):
        # could be list of tuples like [[0],[2],...]
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
            return [int(i) for i in idxs]
    # fallback single int
    try:
        return [int(idxs)]
    except:
        return []

# ---------- Main loop ----------
def main():
    global GLOBAL_FOCAL
    cam = ThreadedCam(src=CAM_ID, width=640, height=480)
    last_red_state = False
    last_red_time = 0.0

    print("Starting. Press ESC to quit.")
    print("If you want to calibrate focal for accurate meters: press 'c' while a target is visible and follow console prompt.")

    while True:
        t0 = time.time()
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue

        display = frame.copy()

        # low-light enhance (non-destructive)
        enhanced = enhance_lowlight(frame, clip_limit=3.0, tile_grid=(8,8), gamma=1.15)

        # prepare input for ONNX (model expects normalized 640x640)
        img = cv2.resize(enhanced, (INPUT_SIZE, INPUT_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(img_rgb, (2,0,1))[None, :, :, :]

        outputs = session.run(None, {input_name: inp})
        boxes_norm, scores, cids = parse_yolov8_output(outputs[0], conf_thr=CONF_THRESHOLD)

        # convert normalized boxes to pixel rects on original display
        rects = []
        info = []
        for (xc,yc,wn,hn), score, cid in zip(boxes_norm, scores, cids):
            x_px = int((xc - wn/2) * display.shape[1])
            y_px = int((yc - hn/2) * display.shape[0])
            w_px = int(wn * display.shape[1])
            h_px = int(hn * display.shape[0])
            # clamp
            x_px = max(0, min(display.shape[1]-1, x_px))
            y_px = max(0, min(display.shape[0]-1, y_px))
            w_px = max(2, min(display.shape[1]-x_px, w_px))
            h_px = max(2, min(display.shape[0]-y_px, h_px))
            rects.append([x_px, y_px, w_px, h_px])
            info.append((cid, score))

        # NMS
        keep = []
        if len(rects) > 0:
            confidences = [float(s) for (_, s) in info]
            idxs = cv2.dnn.NMSBoxes(rects, confidences, CONF_THRESHOLD, NMS_IOU)
            keep = unify_nms_indices(idxs)
        else:
            keep = []

        # draw boxes, compute distances, alerts
        any_red = False
        det_count = 0
        nearest = None

        for i in keep:
            if i < 0 or i >= len(rects):
                continue
            x,y,w_box,h_box = rects[i]
            cid, conf = info[i]
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class{cid}"

            det_count += 1

            # choose pixel width for distance (use bbox width; more stable for wide objects)
            pixel_width = w_box
            # if calibration for this class exists, use it; else GLOBAL_FOCAL
            focal = GLOBAL_FOCAL
            if label in focal_by_class:
                try:
                    fv = focal_by_class[label]
                    focal = float(fv['focal']) if isinstance(fv, dict) else float(fv)
                except:
                    focal = GLOBAL_FOCAL

            # compute raw distance (meters)
            dist = estimate_distance_from_width(pixel_width, KNOWN_WIDTH, focal)
            # smooth by approximate center key
            cx = x + w_box/2
            cy = y + h_box/2
            dist_s = smooth_distance_for_bbox(cx, cy, dist)

            if nearest is None or dist_s < nearest:
                nearest = dist_s

            # determine alert color
            if dist_s <= RED_THRESH:
                color = (0,0,255)
                status = "RED - DANGER"
                any_red = True
            elif dist_s <= YELLOW_THRESH:
                color = (0,255,255)
                status = "YELLOW - CAUTION"
            else:
                color = (0,255,0)
                status = "GREEN - SAFE"

            # draw
            cv2.rectangle(display, (x,y), (x+w_box, y+h_box), color, 2)
            cv2.putText(display, f"{label} {conf:.2f}", (x, max(12,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # distance label
            if dist_s < 999:
                cv2.putText(display, f"{dist_s:.2f} m", (x, y+h_box+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, status, (x, y+h_box+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # beep only when entering red zone (not repeated)
        if any_red and (not last_red_state or (time.time() - last_red_time) > 1.0):
            threading.Thread(target=lambda: winsound.Beep(1400, 300), daemon=True).start()
            last_red_time = time.time()
            last_red_state = True
        elif not any_red:
            last_red_state = False

        # HUD (FPS, detections, nearest distance, mode, time)
        fps = 1.0 / (time.time() - t0 + 1e-8)
        cv2.rectangle(display, (8,8), (380,118), (16,16,16), -1)
        cv2.putText(display, f"FPS: {fps:.1f}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,128), 2)
        cv2.putText(display, f"Detections: {det_count}", (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        if nearest is not None:
            cv2.putText(display, f"Nearest: {nearest:.2f} m", (14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(display, f"Mode: Low-light Detection", (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.putText(display, datetime.now().strftime("%H:%M:%S"), (display.shape[1]-110, display.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("AI Pedestrian Detection - Low Light (ESC to quit, 'c' to calibrate)", display)
        key = cv2.waitKey(1) & 0xFF

        # Interactive calibration: press 'c'
        if key == ord('c'):
            # pick largest detection to calibrate (by area) among kept rects
            if len(keep) == 0:
                print("No detections to calibrate. Ensure target is visible and press 'c' again.")
            else:
                areas = [(rects[i][2]*rects[i][3], i) for i in keep]
                areas.sort(reverse=True)
                idx = areas[0][1]
                pixel_w = rects[idx][2]
                print(f"Captured pixel width = {pixel_w}. Enter measured distance in meters (e.g., 2.0) and press Enter:")
                try:
                    s = input().strip()
                    dist_m = float(s)
                    # compute focal: focal = (pixel_width * distance) / known_width
                    f_est = (pixel_w * dist_m) / KNOWN_WIDTH
                    print(f"Estimated focal: {f_est:.2f} px. Saving to {FOCAL_FILE}.")
                    focal_by_class['person'] = {"focal": f_est}
                    known_heights['person'] = DEFAULT_REAL_HEIGHT
                    with open(FOCAL_FILE, "w") as f:
                        json.dump({"focal_by_class": focal_by_class, "KNOWN_HEIGHTS": known_heights}, f, indent=2)
                    GLOBAL_FOCAL = f_est
                    print("Calibration saved. Distances will be more accurate now.")
                except Exception as e:
                    print("Calibration failed:", e)

        if key == 27:
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # make last_red_state/time global for beep handling
    global last_red_state, last_red_time
    last_red_state = False
    last_red_time = 0.0
    main()
1