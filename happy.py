import cv2
import streamlit as st
from deepface import DeepFace
import tempfile
import os
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Video Facial Expression Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Facial Expression Recognition from Video")
st.write("""
Upload a video file, and the application will analyze it frame by frame. 
After processing, a summary of all detected emotions will be displayed as a table and a graph.
""")

# --- Sidebar with Instructions ---
with st.sidebar:
    st.header("Instructions")
    st.info(
        "This app uses `deepface` for analysis, `OpenCV` for video processing, "
        "`pandas` for data handling, and `Streamlit` with `Plotly` for the interface."
    )
    st.header("How to Use:")
    st.markdown("""
    1.  **Upload a Video:** Use the file uploader to select a video (e.g., MP4, AVI, MOV).
    2.  **Start Analysis:** Click the **"Analyze Video"** button.
    3.  **View Results:** The video will play in the main panel. Real-time analysis appears on the right.
    4.  **Review Summary:** Once the video is fully processed, an overall summary and a bar chart will appear in the right-hand column.
    """)
    st.header("Notes")
    st.markdown("""
    - Processing can be slow for long videos.
    - The model is most accurate with clear, well-lit, front-facing subjects.
    """)

# --- Initialize Session State ---
# This helps maintain state across reruns
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'emotion_stats' not in st.session_state:
    st.session_state.emotion_stats = {}

# --- Main Application Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Video Input")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])
    FRAME_WINDOW = st.image([])

with col2:
    st.header("Analysis Results")
    # Placeholders for real-time updates
    emotion_log = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    # Placeholder for the final summary graph
    summary_chart = st.empty()


if uploaded_file is not None:
    # Use a temporary file to handle the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    if st.button("Analyze Video"):
        st.session_state.analysis_complete = False
        st.session_state.emotion_stats = {}
        summary_chart.empty() # Clear previous charts

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
                            region = analysis[0]['region']
                            x, y, w, h = region['x'], region['y'], region['w'], region['h']

                            # Draw rectangle and label on frame
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"Emotion: {dominant_emotion}"
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Update emotion log and stats
                            emotion_log.markdown(f"**Current Frame Emotion:**\n- **{dominant_emotion}**")
                            emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
                        else:
                            emotion_log.markdown("**Current Frame Emotion:**\n- No face detected")

                    except Exception as e:
                        emotion_log.markdown(f"**Current Frame Emotion:**\n- Error analyzing frame")

                    # Display the processed frame
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    progress_bar.progress(frame_count / total_frames)
                
                # Store results in session state and set completion flag
                st.session_state.emotion_stats = emotion_counts
                st.session_state.analysis_complete = True
                cap.release()
        finally:
            os.remove(video_path) # Clean up the temporary file

# --- Display Final Summary and Graph ---
if st.session_state.analysis_complete:
    with col2:
        st.header("Overall Analysis Summary")
        stats = st.session_state.emotion_stats

        if not stats:
            st.write("No emotions were detected in the video.")
        else:
            # Create a DataFrame from the emotion stats
            emotion_df = pd.DataFrame(stats.items(), columns=['Emotion', 'Frame Count'])
            emotion_df = emotion_df.sort_values(by='Frame Count', ascending=False)

            # Display the data table
            st.subheader("Emotion Counts")
            st.dataframe(emotion_df)

            # Display the bar chart using Plotly
            st.subheader("Emotion Distribution")
            fig = px.bar(
                emotion_df,
                x='Emotion',
                y='Frame Count',
                color='Emotion',
                title="Emotion Distribution in Video"
            )
            summary_chart.plotly_chart(fig, use_container_width=True)

else:
    if uploaded_file is None:
        # This is the modified line of code
        st.info("Welcome! To get started, use the 'Choose a video file...' button above to upload your video.")