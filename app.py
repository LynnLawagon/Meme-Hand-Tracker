from flask import Flask, Response, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

app = Flask(__name__)

# SETTINGS
BASE_MEME_PATH = "static/img/monekey"

camera_enabled = True
hands_detected = 0

current_meme = "normal"
current_meme_path = "/static/img/monekey/normal.png"

# LOAD MODEL
model = None
labels = []

if os.path.exists("models/keras_model.h5"):
    model = tf.keras.models.load_model("models/keras_model.h5")

if os.path.exists("models/labels.txt"):
    with open("models/labels.txt") as f:
        for line in f:
            parts = line.strip().split(" ",1)
            labels.append(parts[1] if len(parts)>1 else parts[0])

# MEME FILES
meme_dict = {"normal":"/static/img/monekey/normal.png"}

meme_path = Path(BASE_MEME_PATH)

for f in meme_path.glob("*.*"):
    meme_dict[f.stem] = f"/static/img/monekey/{f.name}"

# MEDIAPIPE
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# CAMERA
cap = cv2.VideoCapture(0)

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

# VIDEO STREAM
def gen_frames():

    global current_meme
    global current_meme_path
    global hands_detected

    while True:

        if not camera_enabled:

            black = np.zeros((360,640,3),dtype=np.uint8)

            cv2.putText(
                black,
                "CAMERA OFF",
                (200,180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )

            _,buffer=cv2.imencode(".jpg",black)

            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n'+
                  buffer.tobytes()+b'\r\n')

            continue

        success,frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(640,360))

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hands_detected = len(results.multi_hand_landmarks)

            for hand in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                h,w,_ = frame.shape

                xs=[p.x*w for p in hand.landmark]
                ys=[p.y*h for p in hand.landmark]

                x1=max(0,int(min(xs))-30)
                y1=max(0,int(min(ys))-30)
                x2=min(w,int(max(xs))+30)
                y2=min(h,int(max(ys))+30)

                crop = frame[y1:y2,x1:x2]

                gesture = predict(crop)

                if gesture and gesture in meme_dict:

                    current_meme = gesture
                    current_meme_path = meme_dict[gesture]

        else:

            hands_detected = 0
            current_meme = "normal"
            current_meme_path = meme_dict["normal"]

        _,buffer=cv2.imencode(".jpg",frame)

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+
              buffer.tobytes()+b'\r\n')
        
# ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

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

    return jsonify({
        "enabled":camera_enabled
    })

@app.route("/reset")
def reset():

    global current_meme,current_meme_path

    current_meme="normal"
    current_meme_path=meme_dict["normal"]

    return jsonify({"status":"reset"})

# START
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)