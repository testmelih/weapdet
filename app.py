import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('best.onnx')  # Replace with your model path

st.title("Web-Based Live Camera Weapon Detection")
st.write("This app can be accessed on both desktop and mobile devices with a live camera feed.")

# Define a custom video transformer for WebRTC
class YOLOv8Transformer(VideoTransformerBase):
    def __init__(self):
        # Load the YOLO model
        self.model = model

    def transform(self, frame):
        # Convert frame to a format compatible with PIL
        img = frame.to_ndarray(format="bgr24")

        # Convert OpenCV BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = self.model.predict(img_rgb)
        
        # Draw bounding boxes on the image
        annotated_img = results[0].plot()
        
        # Convert RGB image back to BGR for OpenCV display
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
        return annotated_img_bgr

# Setup WebRTC streamer for live video
webrtc_streamer(
    key="weapon-detection",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=YOLOv8Transformer,
    media_stream_constraints={"video": True, "audio": False},
)
