#  Grape Leaf Disease Detection Using Machine Learning

A deep learning-based web application that detects grape leaf diseases from images using **CNN and MobileNetV2**. The project also uses **Grad-CAM** to visualize the regions of the leaf that influence the model's prediction.

##  Overview

Grape leaf diseases can significantly affect crop quality and production. This project provides an automated image-based disease detection system that helps identify grape leaf diseases using deep learning.

The model classifies grape leaves into four categories:

*  Healthy
*  Black Rot
*  Esca
*  Leaf Blight

The trained model achieved approximately **94% classification accuracy** during evaluation.

##  Features

* Image-based grape leaf disease detection
* CNN-based deep learning model
* MobileNetV2 transfer learning
* ~94% model accuracy
* Grad-CAM visualization
* Disease prediction with confidence score
* Flask web application
* MySQL database integration
* Prediction/report generation

##  Technologies Used

| Category      | Technologies                  |
| ------------- | ----------------------------- |
| Language      | Python                        |
| Deep Learning | TensorFlow, Keras             |
| Model         | MobileNetV2, CNN              |
| Visualization | Grad-CAM                      |
| Backend       | Flask                         |
| Database      | MySQL                         |
| Frontend      | HTML, CSS, Bootstrap          |
| Tools         | Jupyter Notebook, Git, GitHub |

##  System Workflow

```text
Grape Leaf Image
       ↓
Image Preprocessing
       ↓
MobileNetV2 CNN Model
       ↓
Feature Extraction
       ↓
Disease Classification
       ↓
Prediction + Confidence
       ↓
Grad-CAM Visualization
       ↓
Result / Report
```

## 🦠 Disease Classes

```text
1. Healthy
2. Black Rot
3. Esca
4. Leaf Blight
```

##  Grad-CAM

Grad-CAM is used to improve model interpretability by highlighting the areas of the grape leaf that contributed to the prediction.

```text
Input Image
     ↓
Trained CNN Model
     ↓
Disease Prediction
     ↓
Grad-CAM
     ↓
Heatmap of Important Regions
```

## 📊 Model

**Architecture:** MobileNetV2
**Type:** Convolutional Neural Network
**Classes:** 4
**Accuracy:** ~94%
**Input:** Grape leaf image

MobileNetV2 was selected because it provides a relatively lightweight architecture while maintaining strong image classification performance.



## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/grape-leaf-disease-detection.git
```

```bash
cd grape-leaf-disease-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🗄️ Database Setup

Create a MySQL database:

```sql
CREATE DATABASE grape_disease_db;
```

Configure your MySQL credentials in the Flask application or, preferably, through environment variables.

Example:

```text
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=grape_disease_db
```

## ▶️ Run the Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000/
```

## 📈 Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Grad-CAM visualization

## 👨‍💻 Author

**Sangamesh**

MCA Graduate | Python | Machine Learning | Deep Learning | Flask | AWS

##  Project Highlights

```text
✓ Machine Learning
✓ Deep Learning
✓ CNN
✓ MobileNetV2
✓ Grad-CAM
✓ TensorFlow
✓ Keras
✓ Flask
✓ MySQL
✓ Computer Vision
```
