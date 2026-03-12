import os, json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/test"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Data generators (augmentation for train)
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Train class indices:", train_gen.class_indices)

# Save class mapping for GUI to interpret predictions reliably
# flow_from_directory gives mapping like {'dented':0, 'not_dented':1} (alphabetical)
mapping = {v:k for k,v in train_gen.class_indices.items()}  # index->label
with open(os.path.join(MODEL_DIR, "class_mapping.json"), "w") as f:
    json.dump(mapping, f)
print("Saved class mapping to models/class_mapping.json ->", mapping)


# Build model (MobileNetV2 backbone)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)   # binary

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# Callbacks

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = os.path.join(MODEL_DIR, f"dent_model_{ts}.keras")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, verbose=1),
    ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
]


# Train

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

import pickle

# SAVE TRAINING HISTORY (IMPORTANT FOR GRAPH)
with open(os.path.join(MODEL_DIR, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training history saved")


# Save final model copy (timestamped)
final_path = os.path.join(MODEL_DIR, f"dent_model_final_{ts}.keras")
model.save(final_path)
print("Training complete. Final model saved to:", final_path)
print("Also saved checkpoint(s) to", MODEL_DIR)