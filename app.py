import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
import webbrowser
import os

# Try to load model with error handling
@st.cache_resource
def load_emotion_model():
    try:
        from keras.models import load_model
        model = load_model("models/model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure the model file exists and is compatible with your TensorFlow version.")
        return None

@st.cache_data
def load_labels():
    try:
        if os.path.exists("models/labels.npy"):
            return np.load("models/labels.npy")
        else:
            st.error("Labels file not found. Please ensure models/labels.npy exists.")
            return np.array(["happy", "sad", "angry", "neutral"])  # Default labels
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return np.array(["happy", "sad", "angry", "neutral"])  # Default labels

# Load model and labels
model = load_emotion_model()
label = load_labels()
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender Using AI & Deep Learning")

# Show model status
if model is None:
	st.error("⚠️ Model failed to load. Please check the troubleshooting guide below.")
	with st.expander("Troubleshooting"):
		st.write("""
		**Common issues and solutions:**
		
		1. **Model file missing**: Ensure `models/model.h5` exists
		2. **TensorFlow version mismatch**: Try updating TensorFlow:
		   ```
		   pip install --upgrade tensorflow
		   ```
		3. **Keras compatibility**: The model might be from an older Keras version
		4. **First time setup**: You might need to train a model first using `src/data_training.py`
		""")
else:
	st.success("✅ Model loaded successfully!")

# Initialize session state
if "run" not in st.session_state:
	st.session_state["run"] = "true"
if "emotion_detected" not in st.session_state:
	st.session_state["emotion_detected"] = False
if "current_emotion" not in st.session_state:
	st.session_state["current_emotion"] = ""

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Load current emotion
try:
	emotion = np.load("temp/emotion.npy")[0]
	if emotion and emotion.strip():
		st.session_state["current_emotion"] = emotion
		st.session_state["emotion_detected"] = True
	else:
		st.session_state["emotion_detected"] = False
		st.session_state["current_emotion"] = ""
except:
	emotion = ""
	st.session_state["emotion_detected"] = False
	st.session_state["current_emotion"] = ""

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		##############################
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks and model is not None:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			try:
				pred = label[np.argmax(model.predict(lst))]
				print(f"Detected emotion: {pred}")
				
				# Display emotion on frame with better styling
				cv2.putText(frm, f"Emotion: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
				cv2.putText(frm, "Emotion Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
				
				# Save emotion and update session state
				np.save("temp/emotion.npy", np.array([pred]))
				st.session_state["current_emotion"] = pred
				st.session_state["emotion_detected"] = True
				
			except Exception as e:
				cv2.putText(frm, "Model Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				print(f"Prediction error: {e}")
				
		elif res.face_landmarks:
			cv2.putText(frm, "Model Not Loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
		else:
			cv2.putText(frm, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

		# Optional: Draw minimal face detection box instead of full mesh
		if res.face_landmarks:
			# Get face bounding box
			h, w, _ = frm.shape
			face_landmarks = res.face_landmarks.landmark
			x_coords = [landmark.x * w for landmark in face_landmarks]
			y_coords = [landmark.y * h for landmark in face_landmarks]
			
			# Draw simple bounding box
			x_min, x_max = int(min(x_coords)), int(max(x_coords))
			y_min, y_max = int(min(y_coords)), int(max(y_coords))
			cv2.rectangle(frm, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

		# Removed mesh drawing for cleaner interface
		# drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION, ...)
		# drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		# drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


		##############################

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Show current emotion status
if st.session_state["emotion_detected"]:
	st.success(f"🎭 Current detected emotion: **{st.session_state['current_emotion']}**")
else:
	st.info("👤 Please position your face in the camera to detect emotion")

# Input fields
lang = st.text_input("🌍 Enter Language In Which You Want Songs", placeholder="e.g., English, Hindi, Spanish")
singer = st.text_input("🎤 Please Enter The Name Of Singer (Optional)", placeholder="e.g., Taylor Swift, Arijit Singh")

# Emotion detection section
st.subheader("📹 Emotion Detection")

if model is not None and lang:
	if not st.session_state["emotion_detected"]:
		st.info("Position your face in the camera and wait for emotion detection...")
		webrtc_streamer(
			key="emotion_detection", 
			desired_playing_state=True,
			video_processor_factory=EmotionProcessor
		)
	else:
		st.success("Emotion detected! You can now get music recommendations.")
		if st.button("🔄 Detect New Emotion"):
			# Reset emotion detection
			st.session_state["emotion_detected"] = False
			st.session_state["current_emotion"] = ""
			np.save("temp/emotion.npy", np.array([""]))
			st.rerun()

# Music recommendation section
st.subheader("🎵 Music Recommendation")

# Enhanced validation for music recommendation
recommend_btn = st.button("🎶 Recommend me songs", type="primary")

if recommend_btn:
	if not lang:
		st.error("❌ Please enter a language first!")
	elif not st.session_state["emotion_detected"] or not st.session_state["current_emotion"]:
		st.error("❌ Please detect your emotion first using the camera above!")
	elif model is None:
		st.error("❌ Model not loaded. Please check the troubleshooting guide.")
	else:
		# Build search query
		search_query = f"{lang} {st.session_state['current_emotion']} song"
		if singer and singer.strip():
			search_query += f" {singer}"
		
		st.success(f"🎵 Opening YouTube with search: '{search_query}'")
		webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
		
		# Reset emotion for next detection
		st.session_state["emotion_detected"] = False
		st.session_state["current_emotion"] = ""
		np.save("temp/emotion.npy", np.array([""]))
		
		st.info("Emotion reset. You can detect a new emotion for your next recommendation!")