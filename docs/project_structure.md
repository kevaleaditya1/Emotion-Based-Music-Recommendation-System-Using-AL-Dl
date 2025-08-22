# Project Structure

```
Emotion-Based-Music-Recommendation-System/
├── .devcontainer/              # Development container configuration
│   └── devcontainer.json
├── .git/                       # Git repository files
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── data_collection.py     # Script for collecting emotion data
│   ├── data_training.py       # Script for training the emotion model
│   └── inference.py           # Standalone emotion detection script
├── models/                     # Trained models and labels
│   ├── model.h5               # Main trained emotion detection model
│   ├── labels.npy             # Emotion labels for the model
│   ├── model_backup.h5        # Backup model from liveEmoji
│   └── labels_backup.npy      # Backup labels from liveEmoji
├── data/                       # Training data files
│   ├── hello.npy              # Training data for 'hello' emotion
│   ├── happy.npy              # Training data for 'happy' emotion
│   └── suprise.npy            # Training data for 'surprise' emotion
├── temp/                       # Temporary files
│   └── emotion.npy            # Current detected emotion state
├── docs/                       # Documentation
│   ├── project_structure.md   # This file
│   └── paper published cpp.pdf # Research paper
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── packages.txt               # System packages for deployment
└── README.md                  # Project documentation
```

## File Descriptions

### Core Application
- **app.py**: Main Streamlit web application for emotion-based music recommendation
- **requirements.txt**: Python package dependencies
- **packages.txt**: System-level packages needed for deployment

### Source Code (`src/`)
- **data_collection.py**: Collects facial landmark data for training emotion models
- **data_training.py**: Trains the emotion detection model using collected data
- **inference.py**: Standalone script for real-time emotion detection
- **__init__.py**: Makes src a Python package

### Models (`models/`)
- **model.h5**: Primary trained Keras model for emotion detection
- **labels.npy**: Corresponding emotion labels for the model
- Backup files from the original liveEmoji implementation

### Data (`data/`)
- Contains .npy files with training data for different emotions
- Each file contains facial landmark coordinates for specific emotions

### Temporary Files (`temp/`)
- **emotion.npy**: Stores the currently detected emotion for the web app

### Documentation (`docs/`)
- Project documentation and research papers