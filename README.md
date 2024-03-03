# Gesture Recognition Project
## Overview
This project aims to recognize hand gestures using computer vision techniques and machine learning algorithms. The project involves extracting features from hand gesture videos, generating penultimate layers for training and test datasets, and performing gesture recognition using cosine similarity.



## Getting Started
1.- Get python3.10 for env
```
python3.10 -m venv <virtual-environment-name>
```
2.- Create project
```
 mkdir projectA
 cd projectA
 python3.10 -m venv env
```
3.- Clone the repository:
```
git clone https://github.com/TeoOrt/ProjectPart2.git
```
3.- Clone the repository:
```
source env/bin/activate
```
4.- Install dependencies:
```
pip install -r requirements.txt
```
4.- Run the project:
```
python main.py
```

if it fails check that your python virtual enviroment is running
```
which python
```

should look something like this 
```
~/Desktop/ProjectPT2/Project_Part2_SourceCode/env/bin/python
```


## Directory Structure
```
├── cnn_model.h5
├── frameextractor.py
├── helper__lib
│   ├── classifier.py
│   ├── __init__.py
│   ├── label_mapper.py
│   ├── __pycache__
│   │   ├── classifier.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   ├── label_mapper.cpython-310.pyc
│   │   └── training_classifier.cpython-310.pyc
│   └── training_classifier.py
├── hfe.py
├── main.ipynb
├── main.py
├── README.md
├── requirements.txt
└── Results.csv

```
## Dependencies
- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- Pandas
