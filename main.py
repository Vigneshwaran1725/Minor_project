import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Users/vicky/Music/Minor1/venv/Lib/site-packages/pytesseract/pytesseract.py'

# Load the image you want to process
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    return img

# Preprocess the image for better text detection
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

# Detect and recognize text in the image using pytesseract
def detect_text(image):
    # pytesseract is compatible with TensorFlow for text detection and recognition
    text = pytesseract.image_to_string(image)
    return text

# Convert text to speech using gTTS and play the audio
def text_to_speech(text, language='en'):
    if not text:
        print("No text detected.")
        return
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("detected_text.mp3")
    os.system("start detected_text.mp3")  # For Windows; use "open" for macOS and "xdg-open" for Linux

# Main function
def main(image_path):
    try:
        # Load and preprocess the image
        image = load_image(image_path)
        processed_image = preprocess_image(image)

        # Detect and recognize text
        detected_text = detect_text(processed_image)
        print("Detected Text:", detected_text)

        # Convert detected text to speech
        text_to_speech(detected_text)
    except Exception as e:
        print("Error:", e)

# Run the main function with your image path
image_path = './img1.jpg'
main(image_path)
