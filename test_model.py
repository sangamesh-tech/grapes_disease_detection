import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = 'model/grape_model.h5'
TEST_IMAGE = 'static/uploads/gp1.jpeg'  # Replace with a known grape leaf image

CLASS_NAMES = ['Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight', 'Healthy']

def test_prediction():
    try:
        print("Loading model...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")

        print(f"Testing image: {TEST_IMAGE}")
        img = load_img(TEST_IMAGE, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        class_idx = np.argmax(predictions)

        print("Predictions:", predictions)
        print(f"Class: {CLASS_NAMES[class_idx]} | Confidence: {confidence:.2f}")

    except Exception as e:
        print("Error during prediction:", e)

if __name__ == '__main__':
    test_prediction()
