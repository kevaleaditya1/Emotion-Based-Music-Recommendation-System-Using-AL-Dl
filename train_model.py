#!/usr/bin/env python3
"""
Convenience script to train the emotion detection model from the root directory
"""

import os
import sys
import subprocess

def main():
    print("🤖 Training Emotion Detection Model")
    print("=" * 40)
    
    # Check if data directory exists and has data
    if not os.path.exists("data"):
        print("❌ Data directory not found!")
        print("Please collect training data first using: python src/data_collection.py")
        return
    
    data_files = [f for f in os.listdir("data") if f.endswith('.npy')]
    if not data_files:
        print("❌ No training data found in data directory!")
        print("Please collect training data first using: python src/data_collection.py")
        return
    
    print(f"✅ Found {len(data_files)} training data files: {data_files}")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Run training script
    print("\n🚀 Starting model training...")
    try:
        result = subprocess.run([sys.executable, "src/data_training.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            print("Output:", result.stdout)
            print("\nYou can now run the app with: streamlit run app.py")
        else:
            print("❌ Training failed!")
            print("Return code:", result.returncode)
            print("Standard output:", result.stdout)
            print("Error output:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error running training: {e}")

if __name__ == "__main__":
    main()