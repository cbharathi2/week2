E-Waste Image Classifier

Overview

This project implements an image classification system for categorizing e-waste images into one of 10 predefined categories using a fine-tuned EfficientNetV2B0 model. The system is built with TensorFlow/Keras for model training and evaluation, and Gradio for an interactive user interface. The dataset is assumed to be organized in directories for training, validation, and testing.

Features:

#Data Preprocessing: Uses ImageDataGenerator for data augmentation (rotation, zoom, flipping) and normalization.
#Model Architecture: Fine-tuned EfficientNetV2B0 with a custom classification head (GlobalAveragePooling2D, Dropout, Dense).
#Training: Includes early stopping and model checkpointing to save the best model based on validation accuracy.
#Evaluation: Generates classification reports, confusion matrices, and plots for accuracy/loss and class distribution.
#Inference: Provides a Gradio-based web interface for real-time image classification.

Requirements

To run this project, install the required dependencies:
"pip install tensorflow numpy matplotlib seaborn gradio"
Ensure you have Python 3.8+ installed.

Dataset: https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset
Each directory contains 10 subfolders, each represneting ine class of e-waste:
The dataset should be organized in the following structure:
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
├── test/
│   ├── class1/
│   ├── class2/
│   └── ...

✅ Uses of EfficientNetV2B0 as a Transfer Learning Backbone
EfficientNetV2B0 is a highly optimized convolutional neural network architecture developed by Google, and it serves as a powerful backbone for transfer learning tasks, particularly in image classification

Core Libraries:
  #Tensorflow: For deep learning model building and training.
  #Numpy: For numerical operation and array manipulation.
  #Matplotlib.pyplot: For plotting training curves results.

~ import tensorflow as tf  # Core TensorFlow library

~ from tensorflow.keras import layers, models, optimizers, callbacks  # Layers, model creation, optimizers, and training callbacks

~ from tensorflow.keras.models import Sequential, load_model  # For sequential model architecture and loading saved models

~ from tensorflow.keras.applications import EfficientNetV2B0  # Pretrained EfficientNetV2B0 model for transfer learning

 ~ from tensorflow.keras.applications.efficientnet import preprocess_input  # Preprocessing function specific to EfficientNet

~ import numpy as np  # Numerical operations and array handling

~ import matplotlib.pyplot as plt  # Plotting graphs and images

~ import seaborn as sns  # Plotting graphs and images

~ from sklearn.metrics import confusion_matrix, classification_report  # Evaluation metrics for classification models

~ import gradio as gr  # Web interface library to deploy and test ML models

~ from PIL import Image  # For image file loading and basic image operations

## 1.  Explore and Understand the Data
- Load image dataset using tools.
- Visualize sample images from each class.
- Check the number of images per class to ensure balance.
- Understand image dimensions, color channels, and class labels.

  ### Split data into training, validation, and testing sets.


