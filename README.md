# Face Emotion Detection

This repository contains a project for face emotion detection from scratch using TensorFlow. It includes two Jupyter notebooks: one for training a sequential model and another for deploying the trained model using Tkinter.

## Features
- Import and preprocess facial emotion data
- Train a deep learning model using TensorFlow's Sequential API
- Save and load the trained model
- Deploy the model with a simple Tkinter-based GUI

## Repository Structure
- `train_model.ipynb` - This notebook handles data import, preprocessing, training a sequential model, and saving the trained model.
- `deploy_model.ipynb` - This notebook loads the saved model and provides a Tkinter-based GUI for real-time emotion detection.

## Requirements
To run this project, install the following dependencies:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python tk
```

## Usage

### Training the Model
1. Open `train_model.ipynb` in Jupyter Notebook.
2. Run all the cells to import data, preprocess it, train the model, and save it.

### Deploying the Model
1. Open `deploy_model.ipynb` in Jupyter Notebook.
2. Run all the cells to load the trained model and launch the Tkinter-based application.

## Acknowledgments
This project was built using TensorFlow and Tkinter for deep learning and GUI-based deployment, respectively.

Feel free to contribute or raise issues if you have suggestions or improvements!

