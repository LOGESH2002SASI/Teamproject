from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import filedialog, Tk, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import serial
import time

# TensorFlow GPU configuration
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Load the model
model = load_model('keras_model.h5', compile=False)

# ---------------------- Serial Communication ----------------------
arduino_port = 'COM13'  # Change this to match your system (e.g., "COM3", "/dev/ttyUSB0")
baud_rate = 9600       # Make sure it matches the Arduino code (Serial.begin(9600))

def send_to_arduino(value):
    try:
        with serial.Serial(arduino_port, baud_rate, timeout=2) as ser:
            time.sleep(2)  # Give Arduino time to reset
            ser.write(value.encode())
            print(f"Sent to Arduino: {value}")
    except serial.SerialException as e:
        print(f"Failed to send to Arduino: {e}")


# Disease solutions dictionary
disease_solutions = {
    "Cataract": "Consult an ophthalmologist for a thorough eye examination. Surgery may be required to remove the cataract.",
    "Diabetic Retinopathy": "Manage blood sugar levels. Consult an eye doctor for laser treatment or injections if necessary.",
    "Glaucoma": "Visit an ophthalmologist for prescribed eye drops. Surgery may be recommended in advanced stages.",
    "Normal": "Your eyes are healthy! Continue regular check-ups and maintain a healthy lifestyle."
}

prediction_to_serial = {
    "Cataract": '1',
    "Diabetic Retinopathy": '2',
    "Glaucoma": '3',
    "Normal": '4'
}

# Prediction function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    confidence = np.max(preds)
    preds = np.argmax(preds, axis=1)

    labels = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
    label = labels[preds[0]]

    return label, confidence

# Function to show the image and results
def show_image_and_result(img_path, result, confidence):
    img = Image.open(img_path)
    img = img.resize((400, 400), Image.Resampling.LANCZOS)  # Reduced image size for better layout
    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    result_label.config(text=f"Predicted Result: {result}\nConfidence: {confidence * 100:.2f}%")

    solution = disease_solutions.get(result, "No solution available.")
    solution_label.config(text=f"Suggested Solution: {solution}", wraplength=600)  # Adjusted wraplength

# Function to process the image
def process_image(img_path):
    result_label.config(text="Loading...")
    solution_label.config(text="")

    # Perform prediction
    result, confidence = model_predict(img_path, model)

    # Show image and results
    show_image_and_result(img_path, result, confidence)

    serial_value = prediction_to_serial.get(result)
    if serial_value:
        send_to_arduino(serial_value)
        print(f"Prediction: {result}, Confidence: {confidence:.2f}, Sent to Arduino: {serial_value}")

# Function to select an image
def select_image():
    img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    if img_path:
        threading.Thread(target=process_image, args=(img_path,)).start()
    else:
        print("No image selected.")

# Function to transition from welcome screen to main app
def show_main_app():
    welcome_label.destroy()
    frame.place(relx=0.5, rely=0.5, anchor="center")  # Center the frame after the welcome message

# Initialize Tkinter
root = Tk()
root.title("Revolutionizing Vision Wellness")

# Set full screen
root.state('zoomed')

# Load background image
bg_image = Image.open("background.jpg")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
bg_image = bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_tk = ImageTk.PhotoImage(bg_image)

# Set the background
bg_label = Label(root, image=bg_tk)
bg_label.place(relwidth=1, relheight=1)  # Fill the window with the background image

# Welcome Screen
welcome_label = Label(root, text="Welcome to Revolutionizing Vision Wellness", font=("Helvetica", 24, "bold"), bg="#4CAF50", fg="white", pady=20, padx=20)
welcome_label.place(relx=0.5, rely=0.5, anchor="center")

# Main Application Frame
frame = Frame(root, bg="#ffffff", bd=10)

# Button to select an image
select_button = Button(frame, text="Select Image", command=select_image, font=("Helvetica", 16), bg="#4CAF50", fg="white", relief="raised", bd=4)
select_button.grid(row=0, column=0, columnspan=2, pady=10)

# Image display label
img_label = Label(frame, bg="#ffffff")
img_label.grid(row=1, column=0, columnspan=2, pady=10)

# Prediction result label
result_label = Label(frame, text="", font=("Helvetica", 16), bg="#ffffff", fg="#2f4f4f")
result_label.grid(row=2, column=0, columnspan=2, pady=10)

# Suggested solution label
solution_label = Label(frame, text="", font=("Helvetica", 14), bg="#ffffff", fg="#228b22", wraplength=600)
solution_label.grid(row=3, column=0, columnspan=2, pady=10)

# Footer label
footer_label = Label(frame, text="deep learning technology", font=("Helvetica", 12), bg="#ffffff", fg="#808080")
footer_label.grid(row=4, column=0, columnspan=2, pady=10)

# Schedule transition from welcome screen to main app after 3 seconds
root.after(3000, show_main_app)

# Run the application
root.mainloop()
