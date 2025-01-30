import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
import io
from PIL import Image
import tensorflow as tf

# Suppress TensorFlow and oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')  # Suppresses most TensorFlow logs

# Set Streamlit page config for theme
st.set_page_config(page_title="Real-Time Emotion Detection", page_icon="😊", layout="wide")

# Custom CSS for background styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(to right,rgb(95, 159, 255),rgb(123, 223, 254));
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("😊 Real-Time Emotion Detection")

# Initialize webcam using OpenCV
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    cap.release()
    st.stop()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
frame_placeholder = st.empty()
stop_button = st.button("Stop Camera", key="stop_button")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            st.error(f"Error detecting emotion: {e}")

    # Convert to PIL format for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    frame_placeholder.image(image, caption="Live Emotion Detection", use_container_width=True)

    if stop_button:
        break

cap.release()
st.write("✅ Camera released. Application terminated.")
