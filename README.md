# Gesture Recognition Project
## Overview
This project aims to recognize hand gestures using computer vision techniques and machine learning algorithms. The project involves extracting features from hand gesture videos, generating penultimate layers for training and test datasets, and performing gesture recognition using cosine similarity.

## Getting Started
1.- Clone the repository:
```
git clone https://github.com/yourusername/gesture-recognition.git
```
2.- Install dependencies:
```
pip install -r requirements.txt
```
3.- Run the project:
```
python main.py
```

## Directory Structure
```
gesture-recognition/
│
├── data/
│   ├── training_videos/
│   └── test_videos/
├── src/
│   ├── feature_extractor.py
│   ├── classifier.py
│   └── main.py
├── results/
│   └── Results.csv
└── README.md
```
## Dependencies
- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- Pandas
