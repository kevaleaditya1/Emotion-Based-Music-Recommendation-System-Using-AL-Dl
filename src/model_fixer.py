#!/usr/bin/env python3
"""
Model compatibility fixer for Emotion-Based Music Recommendation System
This script helps fix model loading issues by rebuilding the model with current TensorFlow version
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

def check_model_compatibility():
    """Check if the current model can be loaded"""
    try:
        model = tf.keras.models.load_model("../models/model.h5")
        print("✅ Model loads successfully!")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def rebuild_model_from_data():
    """Rebuild the model using available training data"""
    print("Attempting to rebuild model from training data...")
    
    data_dir = "../data"
    if not os.path.exists(data_dir):
        print("❌ Data directory not found. Cannot rebuild model.")
        return False
    
    # Check for data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if not data_files:
        print("❌ No training data found. Cannot rebuild model.")
        return False
    
    print(f"Found {len(data_files)} data files: {data_files}")
    
    # Load and combine data
    is_init = False
    size = -1
    label = []
    dictionary = {}
    c = 0
    
    for i in data_files:
        file_path = os.path.join(data_dir, i)
        if not is_init:
            is_init = True 
            X = np.load(file_path)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, np.load(file_path)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c+1

    # Convert labels to numeric
    for i in range(y.shape[0]):
        y[i, 0] = dictionary[y[i, 0]]
    y = np.array(y, dtype="int32")
    y = to_categorical(y)

    # Shuffle data
    cnt = np.arange(X.shape[0])
    np.random.shuffle(cnt)
    X = X[cnt]
    y = y[cnt]

    # Build model
    print("Building new model...")
    ip = Input(shape=(X.shape[1],))
    m = Dense(512, activation="relu")(ip)
    m = Dense(256, activation="relu")(m)
    op = Dense(y.shape[1], activation="softmax")(m) 

    model = Model(inputs=ip, outputs=op)
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

    print("Training model...")
    model.fit(X, y, epochs=50, verbose=1)

    # Save model
    os.makedirs("../models", exist_ok=True)
    model.save("../models/model.h5")
    np.save("../models/labels.npy", np.array(label))
    
    print("✅ Model rebuilt and saved successfully!")
    return True

def create_dummy_model():
    """Create a simple dummy model for testing"""
    print("Creating dummy model for testing...")
    
    # Create a simple model
    ip = Input(shape=(1404,))  # Standard face + hand landmarks size
    m = Dense(512, activation="relu")(ip)
    m = Dense(256, activation="relu")(m)
    op = Dense(4, activation="softmax")(m)  # 4 basic emotions
    
    model = Model(inputs=ip, outputs=op)
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
    
    # Create dummy data for training
    X_dummy = np.random.random((100, 1404))
    y_dummy = to_categorical(np.random.randint(0, 4, 100))
    
    # Quick training
    model.fit(X_dummy, y_dummy, epochs=5, verbose=0)
    
    # Save
    os.makedirs("../models", exist_ok=True)
    model.save("../models/model.h5")
    np.save("../models/labels.npy", np.array(["happy", "sad", "angry", "neutral"]))
    
    print("✅ Dummy model created successfully!")
    return True

def main():
    print("🔧 Model Compatibility Fixer")
    print("=" * 40)
    
    # Check current model
    if check_model_compatibility():
        print("No fixes needed!")
        return
    
    print("\n🔄 Attempting to fix model...")
    
    # Try to rebuild from data
    if rebuild_model_from_data():
        if check_model_compatibility():
            print("✅ Model fixed successfully!")
            return
    
    # Last resort: create dummy model
    print("\n🎭 Creating dummy model as fallback...")
    if create_dummy_model():
        if check_model_compatibility():
            print("✅ Dummy model created successfully!")
            print("⚠️  Note: This is a dummy model. For better results, collect training data and retrain.")
            return
    
    print("❌ Unable to fix model. Please check your TensorFlow installation.")

if __name__ == "__main__":
    main()