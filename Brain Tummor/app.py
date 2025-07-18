from flask import Flask, render_template, request
import os
import base64
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Create this folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Models (Replace with your actual paths)
try:
    cnn_model = tf.keras.models.load_model(r'C:\Users\bnava\project\brain tumor\Brain Tummor\model.h5')
    knn_model = joblib.load(r'C:\Users\bnava\project\brain tumor\Brain Tummor\knn_model.pkl')
    label_encoder = joblib.load(r'C:\Users\bnava\project\brain tumor\Brain Tummor\label_encoder.pkl')
    csv_file_path = r"C:\Users\bnava\OneDrive\Documents\Brain Tumor2.csv"
    csv_data = pd.read_csv(csv_file_path)

    # Clean CSV data
    csv_data.columns = csv_data.columns.str.strip()
    if 'Imagefilename' not in csv_data.columns:
        raise KeyError("Column 'Imagefilename' not found in CSV file")
    csv_data['Imagefilename'] = csv_data['Imagefilename'].str.strip()
    csv_data_clean = csv_data.dropna()  # Remove rows with NaN values
except FileNotFoundError as e:
    print(f"Error loading models or CSV: {e}")
    exit()  # Exit if models aren't loaded

# Helper functions
def remove_extension(filename):
    return os.path.splitext(filename)[0].lower()

def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Feature extractor model
feature_extractor = tf.keras.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu')
])

def predict_tumor(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Error preprocessing image."

    tumor_prediction = cnn_model.predict(img)
    tumor_present = tumor_prediction > 0.5

    if tumor_present:
        features = feature_extractor.predict(img)
        tumor_type_encoded = knn_model.predict(features)
        tumor_type = label_encoder.inverse_transform(tumor_type_encoded)[0]

        image_filename = os.path.basename(image_path)
        image_filename_without_extension = remove_extension(image_filename)

        matching_row = csv_data_clean[
            csv_data_clean['Imagefilename'].str.lower() == image_filename_without_extension
        ]

        if not matching_row.empty:
            tumor_name = matching_row['tumor name'].values[0]
            tumor_type_in_csv = matching_row['tumor type'].values[0]
            return f"Yes you have brain tumor , Tumor Name: {tumor_name}, Tumor Type: {tumor_type_in_csv}"
        else:return "No tumor you are healthy "

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    prediction = None
    selected_image = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="No file part")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")
        if file:
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Base64 encode the image for displaying on the frontend
                with open(filepath, "rb") as image_file:
                    selected_image = base64.b64encode(image_file.read()).decode('utf-8')

                prediction = predict_tumor(filepath)
                os.remove(filepath)  # Remove the uploaded file
            except Exception as e:
                prediction = f"An error occurred during prediction: {str(e)}"

    return render_template('index.html', prediction=prediction, selected_image=selected_image)

if __name__ == '__main__':
     app.run(debug=True, port=5001)
