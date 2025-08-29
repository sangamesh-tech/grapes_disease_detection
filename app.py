import os
from datetime import datetime
from io import BytesIO

import cv2
import mysql.connector
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask import send_file
from keras import Model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
# -------------------
# Flask App Setup
# -------------------
app = Flask(__name__)
app.secret_key = 'grape_secret'

# -------------------
# Paths
# -------------------
#UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/grape_model.h5'
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# This line creates the 'static/uploads' folder if it doesn't already exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# -------------------
# Load Model
# -------------------
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ['Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight', 'Healthy']

# -------------------
# MySQL Config
# -------------------
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'grape_disease'
}


# -------------------
# Initialize DB
# -------------------
def init_db():
    db = mysql.connector.connect(**DB_CONFIG)
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            prediction VARCHAR(255),
            confidence FLOAT,
            timestamp DATETIME
        )
    """)
    db.commit()
    cursor.close()
    db.close()


init_db()

# -------------------
# Disease Solutions
# -------------------
DISEASE_SOLUTIONS = {
    "Grape___Black_rot": (
        "• Prune and remove infected leaves and canes during dormant season.\n"
        "• Apply fungicides like Mancozeb, Myclobutanil, or Captan at early stages.\n"
        "• Maintain good air circulation by proper vineyard spacing.\n"
        "• Remove mummified berries and fallen debris from the ground.\n"
        "• Use drip irrigation instead of overhead watering to keep leaves dry."
    ),
    "Grape___Esca": (
        "• Remove and burn infected vines and wood immediately.\n"
        "• Avoid pruning during wet conditions to reduce infection spread.\n"
        "• Use clean, sterilized tools and avoid deep cuts.\n"
        "• Implement preventative fungicide treatments during dormant periods.\n"
        "• Avoid water stress and improve soil drainage conditions."
    ),
    "Grape___Leaf_blight": (
        "• Remove infected leaves and plant debris promptly.\n"
        "• Apply copper-based or sulfur fungicides during wet and humid periods.\n"
        "• Improve airflow by thinning vines and removing excess foliage.\n"
        "• Ensure balanced fertilization to avoid overgrowth.\n"
        "• Monitor frequently, especially after rainfall, and repeat fungicide treatment if necessary."
    ),
    "Healthy": (
        "• Your grapevine is healthy! No action is needed.\n"
        "• Continue regular inspections and monitoring.\n"
        "• Maintain good sanitation and pruning practices.\n"
        "• Apply balanced fertilizer and irrigate properly.\n"
        "• Consider using organic preventative sprays during peak disease seasons."
    )
}


def get_db():
    return mysql.connector.connect(host="localhost", user="root", password="1234", database="grape_disease")

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['email'] = user['email']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

# Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        db = get_db()
        cursor = db.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                       (name, email, password))
        db.commit()
        flash('Registered successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# Dashboard - Different for Admin/User
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session['role'] == 'admin':
        return render_template('admin_dashboard.html')
    else:
        return render_template('user_dashboard.html')

    @app.route('/login')
    def admin_only():
        if session.get('role') != 'admin':
            flash("Access denied!", "danger")
            return redirect(url_for('dashboard'))
        return "Admin feature here"


# -------------------
# Prediction Helper
# -------------------
def predict_disease(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        class_idx = np.argmax(predictions)

        if confidence < 0.70:
            return "Not a grape leaf", confidence
        return CLASS_NAMES[class_idx], confidence
    except Exception as e:
        print("Prediction error:", e)
        return "Prediction Error", 0.0


from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader  # Important: Import ImageReader


def generate_pdf_report(filename, gradcam_filename, prediction, confidence, solution, timestamp):
    """
    Generates a PDF report with text and images, correctly positioning all elements.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # --- Header ---
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 60, "Grape Disease Diagnosis Report")

    # --- Diagnosis Details ---
    # We will track our vertical position with a variable, starting from the top.
    y_position = height - 100

    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, f"Prediction: {prediction}")
    y_position -= 20  # Move down for the next line
    c.drawString(50, y_position, f"Confidence: {round(confidence * 100, 2)}%")
    y_position -= 20  # Move down
    c.drawString(50, y_position, f"Timestamp: {timestamp}")

    # --- Add Images (The Correct Way) ---
    y_position -= 40  # Add some space before the images

    # Define image properties
    img_height = 180
    img_width = 180

    # Draw Original Image (assuming 'filename' is the path)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(85, y_position, "Original Image")
    y_position -= (img_height + 10)  # Move y_position down before drawing the image
    original_img = ImageReader(filename)
    c.drawImage(original_img, 50, y_position, width=img_width, height=img_height, preserveAspectRatio=True)

    # Draw Grad-CAM Image next to it
    c.drawString(320, y_position + img_height + 10, "Grad-CAM Heatmap")  # Draw title above the image
    gradcam_img = ImageReader(gradcam_filename)
    c.drawImage(gradcam_img, 285, y_position, width=img_width, height=img_height, preserveAspectRatio=True)

    # **THIS IS THE KEY FIX:**
    # Update y_position to be below the images we just drew.
    y_position -= 30  # Add a margin below the images

    # --- Solution Text (Now positioned correctly) ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Solution:")
    y_position -= 20  # Move down to start the text block

    # Use a text object for multi-line solutions
    c.setFont("Helvetica", 12)
    text = c.beginText(50, y_position)
    text.setLeading(14)  # Set line spacing
    for line in solution.split('\n'):
        text.textLine(line)
    c.drawText(text)

    # --- Finalize PDF ---
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -------------------
# Grad-CAM Generation
# -------------------
def generate_gradcam_heatmap(model, img_path, last_conv_layer_name="Conv_1", output_path=None):
    """
    Generates a Grad-CAM heatmap for the given image and saves it.

    :param model: Trained Keras model
    :param img_path: Path to the input image
    :param last_conv_layer_name: Last convolutional layer name in the model
    :param output_path: Path to save the heatmap image
    :return: Path of saved heatmap image
    """
    try:
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Create a model that maps the input image to the activations of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # Gradient of the loss with respect to the output feature map
        grads = tape.gradient(loss, conv_outputs)

        # Mean intensity of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0].numpy()

        # Multiply each channel by its corresponding gradient
        for i in range(conv_outputs.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i].numpy()

        # Generate heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # Load original image
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (224, 224))

        # Apply heatmap
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        if output_path is None:
            output_path = img_path.replace(".jpg", "_gradcam.jpg").replace(".jpeg", "_gradcam.jpeg")

        cv2.imwrite(output_path, superimposed_img)
        return output_path
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return None

# -------------------
# Routes
# -------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            prediction, confidence = predict_disease(filepath)
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('result.html', filename=filename,
                                   prediction="Prediction Error",
                                   confidence=0.0,
                                   gradcam_filename=None,
                                   solution="No solution available.")

        # Generate Grad-CAM
        gradcam_filename = None
        gradcam_path = filepath.rsplit('.', 1)[0] + "_gradcam.jpg"
        gradcam = generate_gradcam_heatmap(model, filepath, "Conv_1", gradcam_path)
        if gradcam:
            gradcam_filename = os.path.basename(gradcam)

        solution = DISEASE_SOLUTIONS.get(prediction, "No specific solution available.")

        # Save prediction + Grad-CAM to DB
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO history (filename, gradcam_filename, prediction, confidence, timestamp) VALUES (%s, %s, %s, %s, %s)",
            (filename, gradcam_filename, prediction, float(confidence), datetime.now())
        )
        db.commit()
        cursor.close()
        db.close()

        return render_template('result.html',
                               filename=filename,
                               prediction=prediction,
                               confidence=round(confidence * 100, 2),
                               gradcam_filename=gradcam_filename,
                               solution=solution)

    return render_template('diagnosis.html')

@app.route('/history')
def history():
    db = mysql.connector.connect(**DB_CONFIG)
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, filename, prediction, confidence, timestamp FROM history ORDER BY timestamp DESC")
    records = cursor.fetchall()
    cursor.close()
    db.close()
    return render_template('history.html', history=records)
@app.route('/gallery')
def gallery():
    return render_template('gallery.html', images=os.listdir('static/gallery'))


@app.route('/guide')
def guide():
    return render_template('guide.html')


@app.route('/datasets')
def datasets():
    return render_template('datasets.html')


@app.route('/trained_model')
def trained_model():
    return render_template('model_info.html')


import os
from flask import Flask, redirect, url_for, flash, send_file


# Make sure you have 'os' imported at the top of your app.py

# ... (assuming your app and UPLOAD_FOLDER are configured as advised previously)
# For example:
# UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/download_report/<int:history_id>')
def download_report(history_id):
    # This route now correctly builds absolute paths before generating the PDF.
    try:
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM history WHERE id = %s", (history_id,))
        record = cursor.fetchone()
    except mysql.connector.Error as err:
        flash(f"Database error: {err}")
        return redirect(url_for('history'))
    finally:
        if 'db' in locals() and db.is_connected():
            cursor.close()
            db.close()

    if not record:
        flash("Record not found.")
        return redirect(url_for('history'))

    # --- Construct Absolute Paths (FIX for OSError) ---
    base_filename = record['filename']
    base_gradcam_filename = record.get('gradcam_filename')

    full_path_filename = os.path.join(app.config['UPLOAD_FOLDER'], base_filename)
    full_path_gradcam = None
    if base_gradcam_filename:
        full_path_gradcam = os.path.join(app.config['UPLOAD_FOLDER'], base_gradcam_filename)

    # --- Gather remaining data ---
    prediction = record['prediction']
    confidence = record['confidence'] / 100
    timestamp = record['timestamp']
    solution = DISEASE_SOLUTIONS.get(prediction, "No specific solution is available for this diagnosis.")

    # --- Generate and send the PDF ---
    pdf_buffer = generate_pdf_report(
        filename=full_path_filename,
        gradcam_filename=full_path_gradcam,
        prediction=prediction,
        confidence=confidence,
        solution=solution,
        timestamp=timestamp
    )

    return send_file(pdf_buffer, as_attachment=True, download_name=f'Grape_Report_{history_id}.pdf',
                     mimetype='application/pdf')


@app.route('/delete_history/<int:history_id>', methods=['POST'])
def delete_history(history_id):
    try:
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor(dictionary=True)

        # Fetch filenames for uploaded image and Grad-CAM heatmap
        cursor.execute("SELECT filename, gradcam_filename FROM history WHERE id = %s", (history_id,))
        record = cursor.fetchone()

        if record:
            # Delete uploaded image
            if record['filename']:
                file_path = os.path.join(UPLOAD_FOLDER, record['filename'])
                if os.path.exists(file_path):
                    os.remove(file_path)

            # Delete Grad-CAM heatmap
            if record['gradcam_filename']:
                gradcam_path = os.path.join(UPLOAD_FOLDER, record['gradcam_filename'])
                if os.path.exists(gradcam_path):
                    os.remove(gradcam_path)

            # Delete DB record
            cursor.execute("DELETE FROM history WHERE id = %s", (history_id,))
            db.commit()

        cursor.close()
        db.close()
        flash('Record deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting record: {e}', 'danger')

    return redirect(url_for('history'))


@app.route('/delete_all_history', methods=['POST'])
def delete_all_history():
    try:
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor()

        # Fetch all filenames
        cursor.execute("SELECT filename, gradcam_filename FROM history")
        records = cursor.fetchall()

        # Delete all images
        for record in records:
            if record[0]:  # filename
                file_path = os.path.join(UPLOAD_FOLDER, record[0])
                if os.path.exists(file_path):
                    os.remove(file_path)

            if record[1]:  # gradcam_filename
                gradcam_path = os.path.join(UPLOAD_FOLDER, record[1])
                if os.path.exists(gradcam_path):
                    os.remove(gradcam_path)

        # Delete all records from DB
        cursor.execute("DELETE FROM history")
        db.commit()

        cursor.close()
        db.close()
        flash('All history and images deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting all history: {e}', 'danger')

    return redirect(url_for('history'))

@app.context_processor
def inject_now():
     return {'current_year': datetime.now().year}
# -------------------
# Run
# -------------------
if __name__ == '__main__':
    app.run(debug=True)