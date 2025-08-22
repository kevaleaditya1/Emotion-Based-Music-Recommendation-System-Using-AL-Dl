import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 


import os

# Load model and labels with proper paths
# Check if running from src directory or root directory
if os.path.exists("../models"):
    models_dir = "../models"
elif os.path.exists("models"):
    models_dir = "models"
else:
    print("Error: Models directory not found!")
    print("Please train a model first using data_training.py")
    exit(1)

model_path = os.path.join(models_dir, "model.h5")
labels_path = os.path.join(models_dir, "labels.npy")
print(f"Using models directory: {models_dir}")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please train a model first using data_training.py")
    exit(1)

if not os.path.exists(labels_path):
    print(f"Error: Labels file not found at {labels_path}")
    print("Please train a model first using data_training.py")
    exit(1)

print("Loading model and labels...")
model = load_model(model_path)
label = np.load(labels_path)
print(f"Model loaded successfully!")
print(f"Available emotions: {label}")



holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)



while True:
	lst = []

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))


	if res.face_landmarks:
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

		pred = label[np.argmax(model.predict(lst))]

		print(pred)
		cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

		
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break