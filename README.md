# 🎵 Emotion-Based Music Recommendation System

A real-time AI-powered music recommendation system that detects your emotion using your webcam and recommends music that matches your current mood using deep learning and Streamlit.

---

## 🧠 Overview

This system uses facial emotion recognition to understand your current mood and suggest music accordingly. It works by:
- Capturing your face using your webcam.
- Predicting your emotion using a trained deep learning model.
- Aggregating predictions over time for higher accuracy.
- Recommending music that matches the detected emotion.

---

## 🚀 Features

- 🎥 Real-time webcam-based emotion detection
- 🤖 Pre-trained deep learning model for accurate emotion recognition
- 🎶 Emotion-based curated music recommendation
- 🌐 Easy-to-use web interface powered by Streamlit

---

## 📦 Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- Pandas / NumPy
- Streamlit

---

## ⚙️ Installation

### ✅ Prerequisites

- Python 3.7 or higher
- Webcam
- `pip` package manager

### 🔧 Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/kevaleaditya1/Emotion-Based-Music-Recommendation-System-Using-AL-Dl.git
   cd Emotion-Based-Music-Recommendation-System-Using-AL-Dl
2. **Install Dependency**
   ```bash
   pip install -r requirements.txt
3. **Run Project**
    ```bash
    streamlit run app.py
---
1. **Launch the App**  
   The user runs `app.py` which opens a Streamlit web interface.

2. **Capture Emotion**  
   - Click the **"Capture Emotion"** button.
   - The webcam activates and captures live video.
   - A Haar Cascade Classifier detects your face.
   - The detected face is passed to a **CNN model (`model.h5`)** to classify emotions like:
     - Happy
     - Sad
     - Angry
     - Neutral
     - etc.

3. **Recommend a Song**  
   - After emotion is detected, click the **"Recommend Song"** button.
   - The app redirects you to a **YouTube music link** that matches your emotion.
---
## 💡 Future Improvements
- Integrate Spotify or YouTube APIs.

- Improve frontend using Streamlit components or WebRTC.

- Add logging and user analytics. 
---
## Paper Published 

- [Check Here](https://ijircce.com/admin/main/storage/app/pdf/19z8eZ6fMsx5RQrEDZPTwxNrYG8GvTf7r7FhSNxB.pdf)
