import streamlit as st
import cv2
import tempfile
import pandas as pd
import time
from deepface import DeepFace

st.set_page_config(page_title="Video Emotion Detector", layout="wide")

st.title("Emotion Detection from Video")
st.write("Upload a video file to analyze the basic emotions (Happy, Sad, Angry, Excited) detected in faces.")

# --- Define our custom emotion mapping ---
# This dictionary maps the DeepFace outputs to the basic emotions you requested.
emotion_mapping = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'surprise': 'excited'  # We map 'surprise' to 'excited' as it's the closest visual match
}
# Emotions like 'neutral', 'fear', and 'disgust' will be ignored.

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Video")
        st.video(video_path)

    with col2:
        st.subheader("Analysis Controls")
        if st.button("Detect Basic Emotions", key="detect_button"):
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            results_placeholder = st.empty()

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Error: Could not open video file.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    results = []
                    
                    status_text.text(f"Processing {total_frames} frames...")
                    start_time = time.time()

                    for frame_num in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        progress_text = f"Processing frame {frame_num + 1}/{total_frames}"
                        progress_bar.progress((frame_num + 1) / total_frames, text=progress_text)

                        try:
                            analysis_results = DeepFace.analyze(
                                img_path=frame, 
                                actions=['emotion'], 
                                enforce_detection=False,
                                detector_backend='opencv'
                            )
                            
                            for face_data in analysis_results:
                                if face_data:
                                    dominant_emotion = face_data['dominant_emotion']
                                    
                                    # *** NEW: Apply the mapping logic ***
                                    if dominant_emotion in emotion_mapping:
                                        basic_emotion = emotion_mapping[dominant_emotion]
                                        score = face_data['emotion'][dominant_emotion]
                                        results.append({
                                            'frame': frame_num,
                                            'timestamp_sec': frame_num / fps if fps > 0 else 0,
                                            'emotion': basic_emotion, # Store the mapped emotion
                                            'score': score
                                        })
                        except Exception:
                            pass

                    cap.release()
                    
                    processing_time = time.time() - start_time
                    progress_bar.empty()
                    status_text.success(f"Processing complete in {processing_time:.2f} seconds!")

                    if results:
                        df = pd.DataFrame(results)
                        
                        with results_placeholder.container():
                            st.subheader("Overall Analysis")
                            
                            primary_emotion = df['emotion'].mode()[0]
                            st.metric(label="Primary Emotion Detected", value=primary_emotion.capitalize())

                            st.subheader("Basic Emotion Summary")
                            emotion_counts = df['emotion'].value_counts()
                            st.bar_chart(emotion_counts)
                            
                            st.subheader("Detailed Frame-by-Frame Results")
                            st.dataframe(df.style.format({'timestamp_sec': '{:.2f}', 'score': '{:.2f}'}))

                    else:
                        results_placeholder.warning("No faces showing one of the target emotions (happy, sad, angry, excited) were detected.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

else:
    st.info("Please upload a video file to begin.")