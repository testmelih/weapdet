import streamlit as st
import cv2
from ultralytics import YOLO
from pyngrok import ngrok

st.title("Weapon Detection")
st.text("Live feed weapon detection with alerts")

# Load the YOLO model
model = YOLO("/content/best.pt")

# Stream video
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    results = model(frame)
    # Display detections and alert if weapon detected
    for result in results:
        st.image(result.plot())
        if result.label in ["Knife", "Handgun"]:
            st.warning(f"Weapon detected: {result.label}")

camera.release()
