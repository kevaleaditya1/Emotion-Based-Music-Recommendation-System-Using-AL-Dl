#!/usr/bin/env python3
"""
Convenience script to collect emotion data from the root directory
"""

import os
import sys
import subprocess

def main():
    print("📸 Emotion Data Collection")
    print("=" * 30)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("This will help you collect training data for emotion detection.")
    print("Instructions:")
    print("1. Position your face clearly in the camera")
    print("2. Express the emotion you want to collect")
    print("3. The script will collect 100 samples automatically")
    print("4. Press ESC to stop early if needed")
    print()
    
    # Run data collection script
    try:
        subprocess.run([sys.executable, "src/data_collection.py"], cwd=".")
    except KeyboardInterrupt:
        print("\n⏹️  Data collection stopped by user")
    except Exception as e:
        print(f"❌ Error during data collection: {e}")

if __name__ == "__main__":
    main()