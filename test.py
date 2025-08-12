import cv2
import streamlit as st
from deepface import DeepFace
import tempfile
import os
import pandas as pd
import plotly.express as px

#st.set_option('server.maxUploadSize', 800)  # size in MB, increase as needed
# --- Page Configuration ---
st.set_page_config(
    page_title="Video Facial Expression Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Facial Expression Recognition (Streamlit-Only Solution)")
st.write("""
Upload a video file using the button below. After processing, a summary of emotions will be displayed.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Instructions")
    st.info(
        "This is a pure Streamlit application. The upload limit is controlled by the `.streamlit/config.toml` file."
    )
    st.markdown("""
    1.  **Upload a Video:** Click the 'Browse files' button.
    2.  **Wait for Processing:** The app will analyze the video automatically.
    3.  **View Results:** See the video with annotations and the final summary graph.
    """)

# --- Main Application Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Video Input")
    # THIS IS THE KEY WIDGET FOR A PURE STREAMLIT APP
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])
    FRAME_WINDOW = st.image([])

with col2:
    st.header("Analysis Results")
    emotion_log = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    summary_chart = st.empty()

# The logic only runs AFTER a file has been uploaded via the widget above
if uploaded_file is not None:
    # Use a temporary file to handle the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            st.success("Video loaded! Starting analysis...")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            emotion_counts = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    status_text.success(f"Analysis complete! Processed {frame_count} frames.")
                    break
                
                frame_count += 1
                status_text.text(f"Processing frame {frame_count}/{total_frames}...")

                try:
                    # Analyze frame for emotion
                    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                    if isinstance(analysis, list) and len(analysis) > 0:
                        dominant_emotion = analysis[0]['dominant_emotion'].capitalize()
                        emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
                        region = analysis[0]['region']
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"Emotion: {dominant_emotion}"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        emotion_log.markdown(f"**Current Frame Emotion:**\n- **{dominant_emotion}**")
                    else:
                        emotion_log.markdown("**Current Frame Emotion:**\n- No face detected")

                except Exception as e:
                    emotion_log.markdown(f"**Current Frame Emotion:**\n- Error analyzing frame")

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                progress_bar.progress(frame_count / total_frames)
            
            cap.release()

            # --- Display Final Summary and Graph ---
            with col2:
                st.header("Overall Analysis Summary")
                if not emotion_counts:
                    st.write("No emotions were detected.")
                else:
                    emotion_df = pd.DataFrame(emotion_counts.items(), columns=['Emotion', 'Frame Count'])
                    emotion_df = emotion_df.sort_values(by='Frame Count', ascending=False)
                    st.subheader("Emotion Counts")
                    st.dataframe(emotion_df)
                    st.subheader("Emotion Distribution")
                    fig = px.bar(emotion_df, x='Emotion', y='Frame Count', color='Emotion', title="Emotion Distribution")
                    summary_chart.plotly_chart(fig, use_container_width=True)
    finally:
        os.remove(video_path)

else:
    st.info("Welcome! Please use the uploader above to begin analysis.")