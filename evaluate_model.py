import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# PATHS

MODEL_PATH = "models/dent_model_final_20260206_152448.keras"
TEST_DATASET_PATH = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# DATA GENERATOR

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# PREDICTIONS

y_true = test_data.classes

y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# METRICS

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n📊 MODEL EVALUATION RESULTS")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")


# CONFUSION MATRIX

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Dented", "Dented"],
    yticklabels=["Not Dented", "Dented"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Car Dent Identification")
plt.show()
