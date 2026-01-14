import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Smart Store Product Detection",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 1.0, 0.25, 0.05
)

input_mode = st.sidebar.radio(
    "Select Input Source",
    ["Image", "Webcam", "Video"]
)

st.title("🛒 Smart Store Product Detection System")

# -------------------------------
# Helper: YOLO Detection
# -------------------------------
def detect_frame(frame):
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    annotated = results.plot()
    return annotated, results

# ===============================
# IMAGE MODE
# ===============================
if input_mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)

        annotated, results = detect_frame(frame)

        st.image(annotated, caption="Detection Result", use_container_width=True)

# ===============================
# WEBCAM MODE
# ===============================
elif input_mode == "Webcam":
    st.warning("Press STOP in Streamlit to end webcam stream")

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _ = detect_frame(frame)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated, use_container_width=True)

        cap.release()

# ===============================
# VIDEO MODE
# ===============================
elif input_mode == "Video":
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        tfile = f"temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(tfile)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _ = detect_frame(frame)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated, use_container_width=True)

        cap.release()
