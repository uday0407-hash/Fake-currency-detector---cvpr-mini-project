import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to the model
MODEL_PATH = os.path.join(BASE_DIR, 'Fake-currency.keras')

# Try loading the model safely
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model at {MODEL_PATH}")
    st.error(str(e))
    st.stop()  # Stop further execution if model cannot be loaded

# Function to preprocess image for the model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function for prediction
def predict_currency(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("Fake Currency Detection")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")

    if st.button('Detect'):
        st.info('Processing...')
        prediction = predict_currency(image)
        label = 'Fake Currency' if prediction[0][0] > 0.5 else 'Real Currency'
        st.success(f"Prediction: {label}")
