import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv8 model (use the path to your ONNX or other model format)
try:
    model = YOLO('best.onnx', task='detect')  # Specify 'detect' to avoid task guessing issues
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# Set Streamlit page configuration
st.set_page_config(page_title="Weapon Detection App", layout="centered")

st.title("Live Weapon Detection from Camera")
st.write("This app uses YOLOv8 for real-time weapon detection. It works on desktop and mobile devices.")

# Video processor class for Streamlit WebRTC
class YOLOv8Transformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        # Convert frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")
        
        # Perform object detection
        results = self.model.predict(img)
        
        # Draw bounding boxes on the image
        annotated_img = results[0].plot()
        
        return annotated_img

# WebRTC configuration for live video feed
webrtc_streamer(
    key="weapon-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOv8Transformer,
    media_stream_constraints={"video": True, "audio": False},
)

# Instructions for users if they encounter issues
st.markdown(
    """
    **Instructions:**
    - Allow camera access when prompted.
    - If the video feed is not working, please try refreshing the page or using a different browser.
    """
)

# Display model information (optional)
with st.expander("Model Information"):
    st.write("Model: YOLOv8")
    st.write("Tasks: Object Detection")
    st.write("Classes: Gun, Knife")

# Error handling for WebRTC connection
try:
    # Starting the WebRTC streamer (already handled by `webrtc_streamer` above)
    pass
except Exception as e:
    st.error(f"WebRTC error: {e}")
