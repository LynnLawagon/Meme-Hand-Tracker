import os
import base64
import binascii
import importlib
from flask import Flask, Response, jsonify, render_template, request, send_file
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)

# SETTINGS
BASE_MEME_PATH = Path("static/img/monekey")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "lynnangelaclawagon/meme-hand-tracker")
KAGGLE_DOWNLOAD = os.getenv("KAGGLE_DOWNLOAD", "1") == "1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
startup_diagnostics = {
    "kaggle_download_enabled": KAGGLE_DOWNLOAD,
    "kaggle_dataset": KAGGLE_DATASET,
    "kaggle_download_attempted": False,
    "kaggle_download_succeeded": False,
    "kaggle_error": "",
    "meme_source": "none",
    "meme_images_loaded": 0,
    "normal_meme_key": "",
    "normal_meme_path": "",
    "normal_meme_file": ""
}

camera_enabled = True
hands_detected = 0

current_meme = "normal"
current_meme_path = ""

def build_placeholder_meme_image_bytes():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:] = (22, 22, 22)
    cv2.putText(image, "MEME IMAGE MISSING", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 2)
    cv2.putText(image, "Add local assets or enable Kaggle download", (55, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (190, 190, 190), 1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return b""
    return encoded.tobytes()

PLACEHOLDER_MEME_BYTES = build_placeholder_meme_image_bytes()

# LOAD MODEL
model = None
labels = []

class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

if os.path.exists("models/keras_model.h5"):
    model = tf.keras.models.load_model(
        "models/keras_model.h5",
        custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D},
        compile=False
    )

if os.path.exists("models/labels.txt"):
    with open("models/labels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ",1)
            labels.append(parts[1] if len(parts)>1 else parts[0])

def normalize_key(value):
    return value.strip().lower().replace(" ", "_").replace("-", "_")

def collect_image_files(root_dir):
    files = {}
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = file_path.stem.lower()
        keys = {stem, normalize_key(stem)}
        for key in keys:
            files.setdefault(key, file_path.resolve())
    return files

def load_meme_files():
    files = {}
    local_roots = [
        BASE_MEME_PATH,
        Path("static/img/monkey"),
        Path("static/img/meme"),
        Path("static/img")
    ]

    for root in local_roots:
        if root.exists() and root.is_dir():
            root_files = collect_image_files(root)
            for key, path in root_files.items():
                files.setdefault(key, path)

    if files:
        startup_diagnostics["meme_source"] = "local"
        return files

    if not KAGGLE_DOWNLOAD:
        return files

    startup_diagnostics["kaggle_download_attempted"] = True

    try:
        kagglehub = importlib.import_module("kagglehub")
        dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        startup_diagnostics["kaggle_error"] = str(exc)
        return files

    if dataset_root.exists() and dataset_root.is_dir():
        dataset_files = collect_image_files(dataset_root)
        for key, path in dataset_files.items():
            files.setdefault(key, path)

    if files:
        startup_diagnostics["kaggle_download_succeeded"] = True
        startup_diagnostics["meme_source"] = "kaggle"

    return files

meme_files = load_meme_files()

if "normal" not in meme_files and meme_files:
    first_key = sorted(meme_files.keys())[0]
    meme_files["normal"] = meme_files[first_key]

meme_dict = {key: f"/meme/{key}" for key in meme_files}

if "normal" not in meme_dict:
    meme_dict["normal"] = "/meme/normal"

if startup_diagnostics["meme_source"] == "none":
    startup_diagnostics["meme_source"] = "placeholder"

current_meme = "normal"
current_meme_path = meme_dict.get("normal", "")

startup_diagnostics["meme_images_loaded"] = len(meme_files)
startup_diagnostics["normal_meme_key"] = "normal"
startup_diagnostics["normal_meme_path"] = current_meme_path

normal_file_path = meme_files.get("normal")
startup_diagnostics["normal_meme_file"] = str(normal_file_path) if normal_file_path else "generated-placeholder"

# MEDIAPIPE
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = [[int(a), int(b)] for a, b in sorted(mp_hands.HAND_CONNECTIONS)]

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# PREDICT
def predict(crop):

    if model is None or crop.size == 0:
        return None

    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = img.astype(np.float32)/255
    img = np.expand_dims(img,0)

    preds = model.predict(img,verbose=0)[0]

    idx = np.argmax(preds)
    conf = preds[idx]

    if conf > 0.45 and idx < len(labels):
        return labels[idx]

    return None

def set_normal_state():

    global current_meme
    global current_meme_path
    global hands_detected

    hands_detected = 0
    current_meme = "normal"
    current_meme_path = meme_dict.get("normal", "")

def process_frame(frame):

    global current_meme
    global current_meme_path
    global hands_detected

    if frame is None or frame.size == 0:
        set_normal_state()
        return []

    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(640,360))
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks = []

    if results.multi_hand_landmarks:

        hands_detected = len(results.multi_hand_landmarks)
        detected = "normal"

        for hand in results.multi_hand_landmarks:

            landmarks.append([[float(point.x), float(point.y)] for point in hand.landmark])

            h,w,_ = frame.shape

            xs=[p.x*w for p in hand.landmark]
            ys=[p.y*h for p in hand.landmark]

            x1=max(0,int(min(xs))-30)
            y1=max(0,int(min(ys))-30)
            x2=min(w,int(max(xs))+30)
            y2=min(h,int(max(ys))+30)

            crop = frame[y1:y2,x1:x2]
            gesture = predict(crop)

            if gesture:
                gesture_keys = [normalize_key(gesture), gesture.strip().lower()]
                for key in gesture_keys:
                    if key in meme_dict:
                        detected = key
                        break

            if detected != "normal":
                break

        current_meme = detected
        current_meme_path = meme_dict.get(detected, meme_dict.get("normal", ""))

    else:

        set_normal_state()

    return landmarks
        
# ROUTES
@app.route("/")
def home():
    return render_template("index.html", initial_meme_path=current_meme_path)

@app.route("/meme/<meme_key>")
def serve_meme(meme_key):
    global meme_files
    global meme_dict
    global current_meme_path

    key = normalize_key(meme_key)
    image_path = meme_files.get(key)

    if image_path is None:
        image_path = meme_files.get("normal")

    if (image_path is None or not image_path.exists()) and not meme_files:
        reloaded = load_meme_files()

        if "normal" not in reloaded and reloaded:
            first_key = sorted(reloaded.keys())[0]
            reloaded["normal"] = reloaded[first_key]

        if reloaded:
            meme_files = reloaded
            meme_dict = {meme_key_name: f"/meme/{meme_key_name}" for meme_key_name in meme_files}
            if "normal" not in meme_dict:
                meme_dict["normal"] = "/meme/normal"

            startup_diagnostics["meme_images_loaded"] = len(meme_files)
            startup_diagnostics["normal_meme_key"] = "normal"
            startup_diagnostics["normal_meme_path"] = meme_dict.get("normal", "")
            normal_file = meme_files.get("normal")
            startup_diagnostics["normal_meme_file"] = str(normal_file) if normal_file else "generated-placeholder"

            current_meme_path = meme_dict.get(current_meme, meme_dict.get("normal", ""))
            image_path = meme_files.get(key) or meme_files.get("normal")

    if image_path is not None and image_path.exists():
        return send_file(image_path)

    if PLACEHOLDER_MEME_BYTES:
        return Response(PLACEHOLDER_MEME_BYTES, mimetype="image/png")

    return jsonify({"error":"Meme image not found"}), 404

@app.route("/video_feed")
def video_feed():
    return jsonify({"error":"video_feed is not available in browser camera mode"}), 410

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "kaggle_download_succeeded": startup_diagnostics["kaggle_download_succeeded"],
        "meme_images_loaded": startup_diagnostics["meme_images_loaded"],
        "normal_image": {
            "key": startup_diagnostics["normal_meme_key"],
            "url": startup_diagnostics["normal_meme_path"],
            "file": startup_diagnostics["normal_meme_file"]
        },
        "details": {
            "kaggle_download_enabled": startup_diagnostics["kaggle_download_enabled"],
            "kaggle_dataset": startup_diagnostics["kaggle_dataset"],
            "kaggle_download_attempted": startup_diagnostics["kaggle_download_attempted"],
            "kaggle_error": startup_diagnostics["kaggle_error"],
            "meme_source": startup_diagnostics["meme_source"]
        }
    })

@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():

    global current_meme
    global current_meme_path
    global hands_detected

    if not camera_enabled:
        set_normal_state()
        return jsonify({
            "meme": current_meme,
            "meme_path": current_meme_path,
            "hands": hands_detected,
            "enabled": camera_enabled,
            "landmarks": [],
            "connections": HAND_CONNECTIONS
        })

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")

    if not image_data:
        return jsonify({"error":"Missing image"}), 400

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error":"Invalid image payload"}), 400

    frame_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error":"Image decode failed"}), 400

    landmarks = process_frame(frame)

    return jsonify({
        "meme": current_meme,
        "meme_path": current_meme_path,
        "hands": hands_detected,
        "enabled": camera_enabled,
        "landmarks": landmarks,
        "connections": HAND_CONNECTIONS
    })

@app.route("/stats")
def stats():
    return jsonify({
        "meme":current_meme,
        "meme_path":current_meme_path,
        "hands":hands_detected
    })

@app.route("/toggle_camera")
def toggle_camera():

    global camera_enabled

    camera_enabled = not camera_enabled

    if not camera_enabled:
        set_normal_state()

    return jsonify({
        "enabled":camera_enabled
    })

@app.route("/reset")
def reset():

    set_normal_state()

    return jsonify({"status":"reset"})

# START
if __name__ == "__main__":
    app.run(host=HOST,port=PORT,debug=DEBUG)