import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

MODEL_PATH = "model/grape_model.h5"
LABEL_MAP_PATH = "model/label_map.json"
CONFIDENCE_THRESHOLD = 0.70

model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    class_labels = {v: k for k, v in label_map.items()}

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    max_index = np.argmax(predictions)
    confidence = predictions[0][max_index]

    # If confidence is low or class is "Not_Grape_Leaf"
    predicted_class = class_labels[max_index]
    if confidence < CONFIDENCE_THRESHOLD or predicted_class == "Not_Grape_Leaf":
        return "Not a grape leaf", float(confidence)

    return predicted_class, float(confidence)
