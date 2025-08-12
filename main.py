import streamlit as st
import cv2
import tempfile
import pandas as pd
import time
from transformers import pipeline
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Emotion Detector",
    page_icon="😊",
    layout="wide"
)

# --- App Title and Description ---
st.title("Advanced Human Emotion Detector 🧠")
st.write("This version uses a specialized Hugging Face Vision Transformer model and a two-stage detection process for higher accuracy.")

# --- Caching the Models ---
# This is a Streamlit best practice to load the models only once.
@st.cache_resource
def load_models():
    """Loads the emotion classification pipeline and the face detector."""
    # Load the specialized emotion classification model from Hugging Face
    emotion_classifier = pipeline(
        "image-classification",
        model="jonathandinu/face-expression-recognition"
    )
    # Load a pre-trained face detector from OpenCV
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return emotion_classifier, face_detector

st.info("Models are loading... This might take a moment on the first run.")
emotion_classifier, face_detector = load_models()
st.success("Models loaded successfully!")


# --- Emotion Mapping & Heuristic Tuning ---
emotion_mapping = {
    'happy': 'Happy',
    'sad': 'Sad',
    'angry': 'Angry',
    'surprise': 'Excited'
}
# Heuristic to prioritize a smile, which is a very strong signal.
HAPPY_THRESHOLD = 0.6 # We use a 0-1 score now, not percent. 0.6 is a good start.


# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a video file to analyze", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Your Uploaded Video")
        st.video(video_path)

    with col2:
        st.subheader("Analysis Controls & Results")
        if st.button("▶️ Analyze Emotions", key="detect_button", help="Click to start the emotion analysis"):
            progress_bar = st.progress(0, text="Initializing analysis...")
            status_text = st.empty()
            results_placeholder = st.empty()

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Error: Could not open the video file.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    results = []
                    
                    status_text.info(f"Analyzing {total_frames} frames using the new model...")
                    start_time = time.time()

                    for frame_num in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        progress_text = f"Processing frame {frame_num + 1} of {total_frames}"
                        progress_bar.progress((frame_num + 1) / total_frames, text=progress_text)

                        # --- STAGE 1: Face Detection with OpenCV ---
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                        for (x, y, w, h) in faces:
                            # Crop the face from the original color frame
                            face_crop = frame[y:y+h, x:x+w]
                            # Convert the cropped face to a PIL Image, which the Hugging Face model needs
                            pil_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                            
                            # --- STAGE 2: Emotion Classification with Hugging Face ---
                            predictions = emotion_classifier(pil_image)
                            
                            # Convert the list of predictions into a dictionary for easier access
                            emotion_scores = {p['label'].lower(): p['score'] for p in predictions}
                            
                            final_emotion = None
                            
                            # --- Heuristic Logic ---
                            if emotion_scores.get('happy', 0) > HAPPY_THRESHOLD:
                                final_emotion = 'Happy'
                            else:
                                # Find the dominant emotion among the mapped ones
                                top_emotion = max(emotion_scores, key=emotion_scores.get)
                                if top_emotion in emotion_mapping:
                                    final_emotion = emotion_mapping[top_emotion]

                            if final_emotion:
                                results.append({
                                    'Frame': frame_num,
                                    'Timestamp (s)': frame_num / fps if fps > 0 else 0,
                                    'Emotion': final_emotion,
                                })

                    cap.release()
                    processing_time = time.time() - start_time
                    progress_bar.empty()
                    status_text.success(f"Analysis complete in {processing_time:.2f} seconds!")

                    if results:
                        df = pd.DataFrame(results)
                        with results_placeholder.container():
                            st.header("Analysis Summary")
                            primary_emotion = df['Emotion'].mode().iloc[0]
                            st.metric(label="Primary Emotion", value=primary_emotion)
                            st.subheader("Emotion Distribution")
                            emotion_counts = df['Emotion'].value_counts()
                            st.bar_chart(emotion_counts)
                            st.subheader("Detailed Frame-by-Frame Log")
                            st.dataframe(df.style.format({'Timestamp (s)': '{:.2f}'}))
                    else:
                        results_placeholder.warning("No faces were detected or no target emotions were classified.")
            except Exception as e:
                st.error(f"A critical error occurred: {e}")
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
else:
    st.info("Please upload a video file to begin the analysis.")