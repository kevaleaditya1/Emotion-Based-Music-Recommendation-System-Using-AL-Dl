#!/usr/bin/env python3
"""
Setup script for Emotion-Based Music Recommendation System
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import subprocess
import sys

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['temp', 'data', 'models', 'docs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Successfully installed Python requirements")
    except subprocess.CalledProcessError:
        print("Error installing requirements. Please install manually using: pip install -r requirements.txt")

def check_model():
    """Check if model exists and is loadable"""
    if not os.path.exists("models/model.h5"):
        print("⚠️  Model file not found. You may need to train a model first.")
        print("Run 'python src/model_fixer.py' to create a basic model.")
        return False
    
    try:
        import tensorflow as tf
        tf.keras.models.load_model("models/model.h5")
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️  Model loading issue: {e}")
        print("Run 'python src/model_fixer.py' to fix model compatibility.")
        return False

def main():
    print("Setting up Emotion-Based Music Recommendation System...")
    create_directories()
    install_requirements()
    check_model()
    print("\nSetup complete! You can now run the application with:")
    print("streamlit run app.py")
    print("\nIf you encounter model issues, run:")
    print("python src/model_fixer.py")

if __name__ == "__main__":
    main()