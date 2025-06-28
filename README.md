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

  
E- WASTE IMAGE CLASSIFICARTION WORK FLOW
![ChatGPT Image Jun 27, 2025, 02_27_04 PM](https://github.com/user-attachments/assets/f2420671-36ba-4f72-b68d-fcde88402d02)

The model is trained and validated on augmented data. Evaluation includes:

- 📈 Accuracy & loss plots
- 🔍 Classification report
- 🧮 Confusion matrix
- 📦 Model saved as `final_model_saved.keras` and best checkpointed model

 ## 2.Split data into training, validation, and testing sets.

train_data = train_gen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_data = val_test_gen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', shuffle=False)
test_data = val_test_gen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', shuffle=False)

Training: The script trains the model for up to 15 epochs with early stopping (patience=3) and saves the best model to model/best_model.keras.
Evaluation: After training, it generates a classification report, confusion matrix, and plots for accuracy/loss and class distribution.
Model Saving: The final model is saved as model/final_model_saved.keras.
![train](https://github.com/user-attachments/assets/64b39383-9af3-4908-b6b2-e162a453470c)

![valid](https://github.com/user-attachments/assets/fee9afa0-b499-4bb6-88e3-57e49b1f162c)


![test](https://github.com/user-attachments/assets/6dd6c4b7-3a8c-47f4-a612-380e236b7bf3)

## 3. TRAIN THE MODEL
# Set the number of epochs to train the model
epochs = 15

# Train the model on the training dataset 'datatrain'
history = model.fit(
    datatrain,                      
    validation_data=datavalid,     
    epochs=epochs,                  
    batch_size=100,                
    callbacks=[early]               
)


![epoch](https://github.com/user-attachments/assets/87c0d5f0-3f43-42c7-a377-d8505b44ae21)
  
## 4. MODEL [ERFORMANCE VISUALIZATION: ACCURACY & LOSS:
# --- Accuracy and Loss Plot ---
plt.figure(figsize=(12, 5))

# Plotting Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

# Plotting Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

![ACCURACY](https://github.com/user-attachments/assets/c434fe12-0c67-4009-b2a9-60ff45273013)

## 5. Model Evaluation
- Plot training and validation accuracy/loss curves.
- Evaluate model performance on validation or test set.
- Use metrics like:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)
  - `confusion_matrix`, `classification_report`: To evaluate the model's classification performance.
 
    ![CONFUSION MATRIX](https://github.com/user-attachments/assets/a5c3445e-dcca-4d44-90b7-d4355ebc929e)


## 6. FINAL TESTING AND SAVING THE MODEL:

![MODEL](https://github.com/user-attachments/assets/43819f2a-ffe9-4db3-a39a-868f4dd8feb1)

Save th etrained model using model.save() or save_model() for future inference.

# 7. MODEL DEPLOYMENT(OPTIONAL)
# Create a web interface using Gradio.
# Load the saved model and preprocess input image before prediction.

!pip install gradio

🌐 Gradio Interface and Preprocessing
- `gr`: To build a web interface for the model.
- `PIL.Image`: For handling image input in Gradio.
- `preprocess_input`: Preprocessing method for EfficientNet.
- `load_model`: For loading a saved model for inference.

*Running on local URL: http://127.0.0.1:7860

![LINK](https://github.com/user-attachments/assets/d0157873-aa51-4fdb-a323-810db4b4d45a)

The link redirect to the interfce that predicts the E-Waste
![GRADIOPRE](https://github.com/user-attachments/assets/3b7c89d9-0599-47c5-aeee-c966c577c18d)

##8. GRADIO PREDICTS THE E-WASTE:
A web page with an image upload area.
After uploading, it shows:
A bar chart or label view with the top 3 probabilities.
A text output with the final predicted class name.

![GRADIOAFTR](https://github.com/user-attachments/assets/7ef6795e-4261-4c24-8bc2-7d02c22f04cb)

CONCLUSION: 
This project successfully classifies e-waste images into different categories using the EfficientNetV2B0 model with transfer learning. The model achieved good accuracy and performance on test data, and a user-friendly Gradio interface was built for real-time predictions. It provides an efficient and scalable solution for automating e-waste classification, helping to support better recycling and waste management










