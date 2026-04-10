# Meme Hand Tracking System

## About
The Meme Hand Tracking System is a Python computer vision project that uses MediaPipe and OpenCV to detect real-time hand gestures through a webcam and switch meme images based on the detected gesture. It combines gesture recognition and a Flask web interface to create an interactive and fun application controlled by simple hand movements.

## Features
- Real-time hand tracking using webcam  
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
