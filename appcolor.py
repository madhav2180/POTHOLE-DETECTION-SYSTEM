import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyttsx3

# Load the trained model
MODEL_PATH = 'full_model.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary for thread safety

# Input shape expected by the model
INPUT_SHAPE = (300, 300, 3)  # Input shape for color image

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to preprocess image and predict pothole or non-pothole
def predict_pothole(image_path):
    img = Image.open(image_path)
    img = img.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "Non-Pothole"
        voice_alert("")
    else:
        result = "Pothole"
        voice_alert("Pothole detected ahead")

    return img, result

# Function to handle image selection and prediction
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img, result = predict_pothole(file_path)

        # Display the image
        img = Image.fromarray(np.uint8(img))  # Convert NumPy array to Image
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        # Update the result label
        result_label.config(text=f"Prediction: {result}")

# Function for voice alert
def voice_alert(message):
    engine.say(message)
    engine.runAndWait()

# Create the main window
root = tk.Tk()
root.title("Pothole Detection System")

# Set the window dimensions
window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Set background color
root.configure(bg='#E5E5E5')

# Create GUI components with enhanced styling
title_label = tk.Label(root, text="Pothole Detection System", font=("Arial", 24), bg='#E5E5E5')
title_label.pack(pady=20)

select_button = tk.Button(root, text="Select Image", command=select_image, font=("Arial", 14), bg='#FF6347', fg='white')
select_button.pack(pady=10)

image_label = tk.Label(root, bg='#FFFFFF')
image_label.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 18), bg='#E5E5E5')
result_label.pack()

# Start the main loop
root.mainloop()
