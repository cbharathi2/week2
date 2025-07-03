import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import gradio as gr

# --- Paths and Parameters ---
dataset_path = "dataset"
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# --- Data Generators ---
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                               width_shift_range=0.2, height_shift_range=0.2,
                               horizontal_flip=True)
val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_data = val_test_gen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', shuffle=False)
test_data = val_test_gen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', shuffle=False)

class_labels = list(train_data.class_indices.keys())

# --- Class Distribution Plot ---
def plot_distribution(generator, title):
    labels = generator.classes
    plt.figure(figsize=(8, 4))
    sns.countplot(x=labels)
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
    plt.title(f'{title} Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

plot_distribution(train_data, "Train")
plot_distribution(val_data, "Validation")
plot_distribution(test_data, "Test")

# --- Model Building ---
base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# ✅ Enable Fine-tuning
for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(len(class_labels), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile with lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

# --- Callbacks and Training ---
os.makedirs("model", exist_ok=True)
callbacks = [
    EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint("model/best_model.keras", monitor='val_accuracy', save_best_only=True)
]
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

# --- Accuracy and Loss Plot ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# --- Save Model ---
model.save("model/final_model_saved.keras")
print("\n✅ Model saved to 'model/final_model_saved.keras'")

# --- Evaluation ---
y_probs = model.predict(test_data, verbose=1)
y_preds = np.argmax(y_probs, axis=1)
y_true = test_data.classes

print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=class_labels))

cm = confusion_matrix(y_true, y_preds)

# Display correct format for confusion matrix
print("\nConfusion Matrix (Raw Counts):")
np.set_printoptions(linewidth=300)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Gradio Interface ---
loaded_model = load_model("model/final_model_saved.keras")

def classify_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = loaded_model.predict(img_array)[0]
    predicted_label = class_labels[np.argmax(predictions)]
    return {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}, predicted_label

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Predicted Class")],
    title="E-Waste Image Classifier",
    description="Upload an image to classify it into one of the 10 categories."
)

demo.launch()
