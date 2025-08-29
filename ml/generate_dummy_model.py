import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Dummy CNN Model
def create_dummy_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes: Black Rot, Esca, Leaf Blight, Healthy
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Build and save dummy model
dummy_model = create_dummy_model()
dummy_model.save('model/grape_model.h5')
print("Dummy model created at model/grape_model.h5")
