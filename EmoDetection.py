import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
import os

# --- Load models ---
MODEL_PATH = "model.h5"  # your emotion classifier
try:
    model = load_model(MODEL_PATH)
    st.success("✅ Emotion model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading emotion model: {e}")
    model = None

labels = ['happy', 'neutral', 'sad', 'surprise']  # use your model’s labels

# Use OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("🎭 Real-time Face Emotion Detection")

st.markdown("""
This app detects your **facial emotion** live from your webcam using **OpenCV Haar Cascade** for face detection.  
Click **Start Camera** to begin.
""")

# Session state
if "run" not in st.session_state:
    st.session_state["run"] = False

def preprocess_face(face_region):
    """Preprocess face region for emotion prediction"""
    try:
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        reshaped_img = resized[:, :, np.newaxis]      # (48,48,1)
        reshaped_img = reshaped_img[np.newaxis, :, :, :]  # (1,48,48,1)
        face_tensor = reshaped_img.astype(np.float32) / 255.0
        return face_tensor
    except Exception as e:
        st.error(f"Error preprocessing face: {e}")
        return None

def predict_label(face_tensor):
    """Predict emotion from preprocessed face tensor"""
    if model is None:
        return "Model not loaded"
    
    try:
        pred = model.predict(face_tensor, verbose=0)
        emotion_index = int(np.argmax(pred))
        return labels[emotion_index]
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Buttons
start = st.button("▶️ Start Camera")
stop = st.button("⏹️ Stop Camera")

if start:
    st.session_state["run"] = True
if stop:
    st.session_state["run"] = False

frame_placeholder = st.empty()
status_placeholder = st.empty()

if st.session_state["run"]:
    if model is None:
        status_placeholder.error("❌ Emotion model not loaded. Please check your model.h5 file.")
        st.session_state["run"] = False
    else:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            status_placeholder.error("❌ Could not open camera.")
            st.session_state["run"] = False
        else:
            status_placeholder.info("📹 Camera started. Press 'Stop Camera' to end.")

            while st.session_state["run"]:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("Could not read from camera.")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                cur_label = "No face"
                num_faces = len(faces)

                for (x, y, w, h) in faces:
                    face_region = frame[y:y+h, x:x+w]
                    face_tensor = preprocess_face(face_region)
                    
                    if face_tensor is not None:
                        cur_label = predict_label(face_tensor)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (196, 140, 100), 2)
                        
                        # Draw emotion label
                        label_y = y - 10 if y - 10 > 10 else y + h + 20
                        cv2.putText(frame, cur_label, (x, label_y),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

                if num_faces > 0:
                    status_placeholder.success(f"✅ Faces: {num_faces} | Emotion: {cur_label}")
                else:
                    status_placeholder.info("👤 No faces detected")

            cap.release()
            status_placeholder.info("📹 Camera stopped.")
else:
    st.info("👆 Click 'Start Camera' to begin emotion detection")
