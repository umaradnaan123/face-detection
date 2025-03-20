import streamlit as st
import cv2  # Ensure OpenCV is installed via `opencv-python-headless`
import numpy as np
from deepface import DeepFace
from PIL import Image
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')

# Set Streamlit page config
st.set_page_config(page_title="Real-Time Emotion Detection", page_icon="😊", layout="wide")

# Custom CSS for background styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(to right, rgb(95, 159, 255), rgb(123, 223, 254));
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("😊 Real-Time Emotion Detection")

# Check if running on Streamlit Cloud
is_cloud = os.getenv("STREAMLIT_SERVER_PORT") is not None

if is_cloud:
    st.warning("Webcam access is not supported in Streamlit Cloud. Using a static image for testing.")
    use_webcam = False
else:
    use_webcam = True

# Load Haar Cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as e:
    st.error(f"Error loading Haar Cascade: {e}")
    st.stop()

# Initialize webcam or static image
if use_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        st.stop()
else:
    static_image_path = "test_image.jpg"
    if not os.path.exists(static_image_path):
        st.error(f"Static image not found at path: {static_image_path}")
        st.stop()
    frame = cv2.imread(static_image_path)
    if frame is None:
        st.error("Error: Could not load static image.")
        st.stop()

frame_placeholder = st.empty()
stop_button_pressed = False

# Button to stop processing
stop_button = st.button("Stop", key="stop_button")
if stop_button:
    stop_button_pressed = True

while (use_webcam and cap.isOpened() and not stop_button_pressed) or (not use_webcam and not stop_button_pressed):
    if use_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break
    else:
        # Use the static image for cloud environments
        pass

    # Convert frame to grayscale for face detection
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

    # Convert the frame to RGB format for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    frame_placeholder.image(image, caption="Live Emotion Detection", use_container_width=True)

    # Check if the stop button was pressed
    if stop_button:
        stop_button_pressed = True

# Release the webcam if it was used
if use_webcam:
    cap.release()

st.write("✅ Processing terminated.")
