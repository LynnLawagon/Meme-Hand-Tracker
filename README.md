# Meme Hand Tracking System

## About
The Meme Hand Tracking System is a Python computer vision project that uses MediaPipe and OpenCV to detect real-time hand gestures from browser camera frames and switch meme images based on the detected gesture. It combines gesture recognition and a Flask web interface to create an interactive and fun application controlled by simple hand movements.

## Features
- Real-time hand tracking from browser camera input  
- Gesture recognition using MediaPipe  
- Dynamic meme image switching  
- Flask web interface  
- Split-screen display (camera + meme)  

## Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- Flask  
- HTML, CSS, JavaScript  

## Project Structure
Meme Hand Tracking
│
├── app.py
├── requirements.txt
├── venv/
│
├── models
│   ├── keras_model.h5
│   └── labels.txt
│
├── static
│   └── img
│       └── monekey
│
└── templates
    └── index.html

## Installation
Install dependencies
pip install -r requirements.txt

## Run the project
python app.py

## Docker
Build the image:

docker build -t meme-hand-tracker .

Run the container:

docker run --rm -p 5000:5000 meme-hand-tracker

Or use Compose:

docker compose up --build

The app binds to `0.0.0.0` inside the container and reads `HOST`, `PORT`, and `FLASK_DEBUG` from the environment.

Camera behavior in this version:

1. The webcam is opened in the browser with getUserMedia.
2. The front end sends JPEG frames to `/analyze_frame` for gesture inference.
3. No host webcam passthrough is required for Docker or cloud deploys.
4. Meme images load from local assets if available, otherwise from the Kaggle dataset `lynnangelaclawagon/meme-hand-tracker` via `kagglehub`.

Optional environment variables:

1. `KAGGLE_DOWNLOAD=1` enables automatic dataset download fallback (default: `1`).
2. `KAGGLE_DATASET` overrides the dataset slug (default: `lynnangelaclawagon/meme-hand-tracker`).

Health diagnostics endpoint:

1. `GET /health` returns startup diagnostics including:
    - `kaggle_download_succeeded`
    - `meme_images_loaded`
    - `normal_image` (which image is currently used as normal)

## Deployment notes
This app works on cloud platforms like Render because camera capture happens in the client browser. The deployed server only receives frame uploads and returns prediction/state updates. Use HTTPS in deployment so browsers allow camera access.
