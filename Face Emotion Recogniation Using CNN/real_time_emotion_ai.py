import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)

# --- Load CNN model ---
MODEL_PATH = "emotion_cnn_model.h5"

@st.cache_resource
def load_emotion_model():
    model = load_model(MODEL_PATH)
    return model

model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Display model info for debugging ---
st.sidebar.subheader("🧩 Model Information")
st.sidebar.write(f"Input shape: `{model.input_shape}`")
st.sidebar.write(f"Output shape: `{model.output_shape}`")

# --- Streamlit App ---
st.title("🎥 Real-Time Facial Emotion Recognition using CNN + Gemini AI")
st.markdown("Use your webcam to detect facial emotions and generate AI-driven insights with **Gemini**.")

# --- Initialize OpenCV ---
run = st.checkbox('▶️ Start Webcam')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(frame, x, y, w, h):
    """Crop and preprocess face ROI for model input."""
    roi = frame[y:y+h, x:x+w]

    # Detect if model expects color or grayscale
    input_shape = model.input_shape
    if input_shape[-1] == 1:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))
    else:
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)
    return roi

if run:
    st.info("✅ Webcam started. Close the checkbox to stop streaming.")
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("⚠️ Could not access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

        for (x, y, w, h) in faces:
            face_roi = preprocess_face(frame, x, y, w, h)
            prediction = model.predict(face_roi)
            label_index = np.argmax(prediction)
            confidence = np.max(prediction)
            print(confidence)
            print(emotion_labels[label_index])
            label = emotion_labels[label_index] 

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not run:
            break
else:
    st.write("🟢 Click 'Start Webcam' to begin real-time detection.")

camera.release()

# --- Gemini Emotional Insight Generator ---
st.subheader("💬 AI Emotional Insight Generator")
selected_emotion = st.selectbox("Select an emotion to analyze", emotion_labels)

if st.button("✨ Generate Insight"):
    with st.spinner("🧠 Gemini analyzing emotional context..."):
        try:
            model_g = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""
            The detected facial emotion is '{selected_emotion}'.
            As an AI psychologist, explain:
            1. What this emotion typically indicates.
            2. In what kind of real-life scenario it might occur.
            3. A short motivational or calming message suitable for this mood.
            """
            response = model_g.generate_content(prompt)
            st.subheader("🧩 Gemini Emotional Response")
            st.write(response.text)
        except Exception as e:
            st.error(f"❌ Gemini API Error: {e}")
