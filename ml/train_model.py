import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import json

DATASET_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')

if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError("Ensure 'dataset/train' and 'dataset/validation' exist.")

train_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1.0/255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
print(f"Classes: {train_data.class_indices}")

os.makedirs('model', exist_ok=True)
with open('model/label_map.json', 'w') as f:
    json.dump(train_data.class_indices, f)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

model.save('model/grape_model.h5')
print("Model saved at model/grape_model.h5")
