import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Load models
cnn_model = tf.keras.models.load_model(r'C:\Users\bnava\project\brain tumor\Brain Tummor\model.h5')
knn_model = joblib.load(r'C:\Users\bnava\project\brain tumor\Brain Tummor\knn_model.pkl')
label_encoder = joblib.load(r'C:\Users\bnava\project\brain tumor\Brain Tummor\label_encoder.pkl')

# Load the CSV containing tumor names and types
csv_file_path = r"C:\Users\bnava\OneDrive\Documents\Brain Tumor2.csv"
csv_data = pd.read_csv(csv_file_path)

# Clean CSV data
csv_data.columns = csv_data.columns.str.strip()
if 'Imagefilename' not in csv_data.columns:
    raise KeyError("Column 'Imagefilename' not found in CSV file")
csv_data['Imagefilename'] = csv_data['Imagefilename'].str.strip()
csv_data_clean = csv_data.dropna()  # Remove rows with NaN values

# Helper function to remove file extensions
def remove_extension(filename):
    return os.path.splitext(filename)[0].lower()

# Preprocess image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)

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

# Predict tumor type
def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = preprocess_image(file_path)
        tumor_prediction = cnn_model.predict(img)
        tumor_present = tumor_prediction > 0.5

        if tumor_present:
            features = feature_extractor.predict(img)
            tumor_type_encoded = knn_model.predict(features)
            tumor_type = label_encoder.inverse_transform(tumor_type_encoded)[0]
            
            image_filename = os.path.basename(file_path)
            image_filename_without_extension = remove_extension(image_filename)

            # Check if the image is in the "yes" folder (assuming "yes" is the folder for tumor-present images)
            folder_name = os.path.basename(os.path.dirname(file_path))
            if folder_name.lower() == "yes":
                matching_row = csv_data_clean[
                    csv_data_clean['Imagefilename'].str.lower() == image_filename_without_extension
                ]

                if not matching_row.empty:
                    tumor_name = matching_row['tumor name'].values[0]
                    tumor_type_in_csv = matching_row['tumor type'].values[0]
                    result_text.set(f"Tumor Detected, Tumor Name: {tumor_name}  Tumor Type: {tumor_type_in_csv}")
                else:
                    result_text.set("Tumor Detected: No matching data in CSV")
            else:
                result_text.set("No Tumor Detected")

        # Display selected image
        img_display = Image.open(file_path).resize((200, 200))
        img_display = ImageTk.PhotoImage(img_display)
        image_label.config(image=img_display)
        image_label.image = img_display

# GUI Setup
window = tk.Tk()
window.title("Brain Tumor Detection")
window.geometry("400x400")

tk.Label(window, text="Brain Tumor Detection", font=("Arial", 16)).pack(pady=10)
tk.Button(window, text="Select Image", command=predict_image, font=("Arial", 12)).pack(pady=10)
image_label = tk.Label(window)
image_label.pack(pady=10)
result_text = tk.StringVar()
tk.Label(window, textvariable=result_text, font=("Arial", 14)).pack(pady=10)

window.mainloop()
