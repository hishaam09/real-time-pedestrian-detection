# cloud_uploader.py
import requests
import threading
import time

# =========== CONFIG - replace these ===========
THINGSPEAK_API_KEY = "ETY1AXR13CSTWZXJ"  # 🔑 replace with your ThingSpeak Write API Key
THINGSPEAK_URL = "https://api.thingspeak.com/update"
THINGSPEAK_INTERVAL = 16.0  # seconds (ThingSpeak free limit)

FIREBASE_DB_URL = "https://pedestriandetection-d663d-default-rtdb.firebaseio.com/"  # 🔑 replace with your Firebase DB URL
FIREBASE_PATH = "/detections/latest.json"
# ===============================================

_last_ts = 0.0
_lock = threading.Lock()

def _send_to_thingspeak(distance, alert_code, count, fps):
    global _last_ts
    now = time.time()
    with _lock:
        if now - _last_ts < THINGSPEAK_INTERVAL:
            return False
        _last_ts = now

    params = {
        "api_key": THINGSPEAK_API_KEY,
        "field1": round(distance, 2),
        "field2": int(alert_code),
        "field3": int(count),
        "field4": round(fps, 2)
    }
    try:
        r = requests.get(THINGSPEAK_URL, params=params, timeout=8)
        return (r.status_code == 200 and r.text != "0")
    except Exception:
        return False

def _send_to_firebase(distance, alert_str, count, fps):
    url = FIREBASE_DB_URL.rstrip("/") + FIREBASE_PATH
    payload = {
        "nearest": round(distance, 2),
        "alert": alert_str,
        "count": int(count),
        "fps": round(fps, 2),
        "timestamp": int(time.time())
    }
    try:
        r = requests.put(url, json=payload, timeout=8)
        return r.status_code in (200, 201)
    except Exception:
        return False

def cloud_send_async(distance, alert_str, count, fps):
    """Non-blocking send to both ThingSpeak & Firebase"""
    alert_code = 1 if alert_str == "red" else (2 if alert_str == "yellow" else 3)
    threading.Thread(target=_send_to_thingspeak, args=(distance, alert_code, count, fps), daemon=True).start()
    threading.Thread(target=_send_to_firebase, args=(distance, alert_str, count, fps), daemon=True).start()
