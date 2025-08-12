import streamlit as st
import cv2
import tempfile
import pandas as pd
import time
from deepface import DeepFace 

st.set_page_config(page_title="Video Emotion Detector", layout="wide")

st.title("Emotion Detection from Video")
st.write("Upload a video file, and this application will analyze the emotions detected in faces, frame by frame. This version uses the DeepFace library.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Use a temporary file to save the uploaded video content
    # This is necessary because OpenCV and DeepFace need a file path to read from
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    # --- Display Video and Run Analysis ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Video")
        st.video(video_path)

    with col2:
        st.subheader("Analysis Controls")
        if st.button("Detect Emotions", key="detect_button"):
            # This block runs when the button is clicked
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            chart_placeholder = st.empty()

            try:
                # Use OpenCV to open the video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Error: Could not open video file.")
                else:
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Store results
                    results = []
                    
                    status_text.text(f"Processing {total_frames} frames...")
                    start_time = time.time()

                    # Loop through each frame of the video
                    for frame_num in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break # Break the loop if there are no more frames
                        
                        # Update progress bar and status text
                        progress_text = f"Processing frame {frame_num + 1}/{total_frames}"
                        progress_bar.progress((frame_num + 1) / total_frames, text=progress_text)

                        # Analyze emotions in the current frame using DeepFace
                        # 'enforce_detection=False' prevents DeepFace from throwing an error if no face is found
                        try:
                            # The result from DeepFace is a list of dictionaries, one for each detected face
                            analysis_results = DeepFace.analyze(
                                img_path=frame, 
                                actions=['emotion'], 
                                enforce_detection=False,
                                detector_backend='opencv' # Using a faster detector
                            )
                            
                            # For each face found in the frame, store the dominant emotion
                            for face_data in analysis_results:
                                if face_data: # Check if the result is not empty
                                    dominant_emotion = face_data['dominant_emotion']
                                    score = face_data['emotion'][dominant_emotion]
                                    results.append({
                                        'frame': frame_num,
                                        'timestamp_sec': frame_num / fps if fps > 0 else 0,
                                        'emotion': dominant_emotion,
                                        'score': score
                                    })
                        except Exception as e:
                            # This can happen if a frame is corrupted, just skip it
                            # st.write(f"Could not analyze frame {frame_num}: {e}") # Optional: for debugging
                            pass

                    # Release the video capture object
                    cap.release()
                    
                    processing_time = time.time() - start_time
                    progress_bar.empty() # Remove the progress bar
                    status_text.success(f"Processing complete in {processing_time:.2f} seconds!")

                    # --- Display Results ---
                    if results:
                        df = pd.DataFrame(results)
                        
                        st.subheader("Emotion Summary")
                        emotion_counts = df['emotion'].value_counts()
                        
                        # Display bar chart in the placeholder
                        with chart_placeholder.container():
                            st.bar_chart(emotion_counts)
                        
                        st.subheader("Detailed Frame-by-Frame Results")
                        st.dataframe(df.style.format({'timestamp_sec': '{:.2f}', 'score': '{:.2f}'}))

                    else:
                        st.warning("No faces were detected in the video.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                # Clean up in case of error
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

else:
    st.info("Please upload a video file to begin.")