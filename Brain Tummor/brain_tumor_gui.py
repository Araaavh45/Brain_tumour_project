import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\bnava\project\brain tumor\Brain Tummor\model.h5')  # Replace with your model's path

def preprocess_image(image_path):
    """Preprocess the input image for prediction."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image():
    """Handle the prediction for the selected image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        
        result_text.set("Brain Tumor" if prediction > 0.5 else "No Brain Tumor")
        
        # Display the selected image
        img_display = Image.open(file_path)
        img_display = img_display.resize((200, 200))
        img_display = ImageTk.PhotoImage(img_display)
        image_label.config(image=img_display)
        image_label.image = img_display

# Set up the GUI window
window = tk.Tk()
window.title("Brain Tumor Detection")
window.geometry("400x400")

# Add components
title_label = tk.Label(window, text="Brain Tumor Detection", font=("Arial", 16))
title_label.pack(pady=10)

select_button = tk.Button(window, text="Select Image", command=predict_image, font=("Arial", 12))
select_button.pack(pady=10)

image_label = tk.Label(window)
image_label.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(window, textvariable=result_text, font=("Arial", 14))
result_label.pack(pady=10)

# Run the application
window.mainloop()



