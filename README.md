# Emotion Extraction from Video

This app lets you upload a 10-second video and detects the dominant emotion (happy, sad, angry, excited) using a pre-trained deep learning model.

## How to use

1. Place a Keras emotion recognition model named `emotion_model.h5` in the workspace root. The model should accept 48x48 grayscale images and output 4 classes: Angry, Happy, Sad, Excited.
2. Install dependencies:
	```bash
	pip install streamlit opencv-python keras tensorflow
	```
3. Run the app:
	```bash
	streamlit run app.py
	```
4. Upload a 10-second video (mp4, avi, mov) and see the detected emotion.

---

**Note:**
- You must provide the model file (`emotion_model.h5`).
- For best results, use videos with clear, front-facing faces.
# video