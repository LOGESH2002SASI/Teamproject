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

# TensorFlow GPU configuration
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Load the model
model = load_model('keras_model.h5', compile=False)

# Disease solutions dictionary
disease_solutions = {
    "Cataract": (
        "A cataract causes clouding of the lens in the eye, leading to blurry or double vision, "
        "faded colors, and difficulty seeing at night. Causes include aging, prolonged UV exposure, "
        "diabetes, smoking, and certain medications.\n\n"
        "Suggested Solution: \n"
        "- Schedule an appointment with an ophthalmologist for a detailed eye examination.\n"
        "- Surgery is the only effective treatment in advanced stages, involving the replacement of the cloudy lens with an artificial one.\n"
        "- Preventive care includes wearing sunglasses, avoiding smoking, and maintaining a healthy diet rich in antioxidants."
    ),
    "Diabetic Retinopathy": (
        "Diabetic retinopathy occurs when high blood sugar damages the blood vessels in the retina, "
        "leading to vision loss or blindness. It progresses through stages, from mild to proliferative retinopathy.\n\n"
        "Suggested Solution: \n"
        "- Strictly monitor and manage blood sugar, blood pressure, and cholesterol levels.\n"
        "- Regular eye exams (at least once a year) to detect early signs of damage.\n"
        "- Advanced cases may require treatments such as laser therapy, anti-VEGF injections to reduce swelling, or vitrectomy surgery."
    ),
    "Glaucoma": (
        "Glaucoma is a group of eye conditions that damage the optic nerve, often due to high intraocular pressure. "
        "Symptoms may include loss of peripheral vision, eye pain, and blurred vision. It can lead to blindness if untreated.\n\n"
        "Suggested Solution: \n"
        "- Visit an ophthalmologist promptly for an accurate diagnosis and to measure eye pressure.\n"
        "- Treatment typically includes prescription eye drops to reduce pressure. Options like laser therapy or surgery may be required in severe cases.\n"
        "- Regular monitoring and lifelong adherence to treatment are essential to prevent further vision loss."
    ),
    "Normal": (
        "Your eyes are healthy! Maintaining good eye health requires consistent care and preventive measures.\n\n"
        "Suggested Solution: \n"
        "- Continue regular eye check-ups (at least once every two years) to detect any emerging issues early.\n"
        "- Protect your eyes by wearing sunglasses with UV protection outdoors.\n"
        "- Reduce eye strain by following the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.\n"
        "- Maintain a balanced diet rich in vitamins A, C, and E, along with omega-3 fatty acids, to support eye health."
    )
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
    img = img.resize((300, 300), Image.Resampling.LANCZOS)  # Resized to fit on the right side
    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    result_label.config(text=f"Predicted Result: {result}\nConfidence: {confidence * 100:.2f}%")

    solution = disease_solutions.get(result, "No solution available.")
    solution_label.config(text=f"{solution}", wraplength=500)  # Adjusted wraplength

# Function to process the image
def process_image(img_path):
    result_label.config(text="Loading...")
    solution_label.config(text="")

    # Perform prediction
    result, confidence = model_predict(img_path, model)

    # Show image and results
    show_image_and_result(img_path, result, confidence)

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

# Prediction result label
result_label = Label(frame, text="", font=("Helvetica", 16), bg="#ffffff", fg="#2f4f4f")
result_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)

# Suggested solution label
solution_label = Label(frame, text="", font=("Helvetica", 14), bg="#ffffff", fg="#228b22", wraplength=500, justify="left")
solution_label.grid(row=2, column=0, sticky="w", padx=10, pady=10)

# Image display label (on the right side)
img_label = Label(frame, bg="#ffffff")
img_label.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="e")

# Footer label
footer_label = Label(frame, text="Powered by Deep Learning Technology", font=("Helvetica", 12), bg="#ffffff", fg="#808080")
footer_label.grid(row=3, column=0, columnspan=2, pady=10)

# Schedule transition from welcome screen to main app after 3 seconds
root.after(3000, show_main_app)

# Run the application
root.mainloop()
