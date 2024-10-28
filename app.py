import streamlit as st
from opencv-python import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = YOLO('best.onnx')  # Replace with the path to your trained model

# Streamlit web app title
st.title("Live Webcam Weapon Detection")

# Start capturing the video from webcam
st.write("Starting video stream...")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

while run:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break

    # Convert the frame (OpenCV format) to PIL image format for YOLOv8 model
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection
    results = model.predict(pil_img)

    # Plot bounding boxes on the frame
    annotated_frame = results[0].plot()  # Annotated frame with bounding boxes

    # Convert frame for displaying in Streamlit
    FRAME_WINDOW.image(annotated_frame)

cap.release()
