# Usage Guide

## Quick Start

1. **Setup the environment**:
   ```bash
   python setup.py
   ```

2. **Run the main application**:
   ```bash
   streamlit run app.py
   ```

## Components

### 1. Web Application (`app.py`)
The main Streamlit application that provides a web interface for:
- Real-time emotion detection using webcam
- Music recommendation based on detected emotion
- User input for language and singer preferences

### 2. Data Collection
Use this script to collect training data for new emotions:
```bash
python collect_data.py
```
Or run directly:
```bash
python src/data_collection.py
```
- Enter the emotion name when prompted
- Look at the camera and express the emotion
- The script will collect 100 samples automatically

### 3. Model Training
Train a new model with collected data:
```bash
python train_model.py
```
Or run directly:
```bash
python src/data_training.py
```
- Automatically finds all .npy files in the data directory
- Trains a neural network model
- Saves the trained model to `models/model.h5`

### 4. Standalone Inference
Run emotion detection without the web interface:
```bash
python src/inference.py
```
- Opens webcam for real-time emotion detection
- Displays detected emotion on screen
- Press ESC to exit

## File Structure

- **models/**: Contains trained models and labels
- **data/**: Training data files (.npy format)
- **temp/**: Temporary files used by the web app
- **src/**: Source code for data collection, training, and inference
- **docs/**: Documentation and research papers

## Troubleshooting

### Model Loading Issues

If you encounter model loading errors:

1. **Run the model fixer**:
   ```bash
   cd src
   python model_fixer.py
   ```

2. **Update TensorFlow**:
   ```bash
   pip install --upgrade tensorflow
   ```

3. **Retrain the model**:
   ```bash
   cd src
   python data_training.py
   ```

### Common Issues

- **"Model file not found"**: Ensure `models/model.h5` exists
- **TensorFlow version mismatch**: Update TensorFlow or use the model fixer
- **Webcam not working**: Check camera permissions and ensure no other app is using it
- **Poor emotion detection**: Ensure good lighting and clear face visibility

## Tips

1. **For better accuracy**: Collect more training data for each emotion
2. **Lighting**: Ensure good lighting when using the webcam
3. **Camera position**: Position your face clearly in the camera frame
4. **Multiple emotions**: You can add new emotions by collecting data and retraining