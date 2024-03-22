import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
#from folium import Map, Circle, Marker  # Import Folium for map creation
#from streamlit_folium import st_folium

# Load your trained models
image_model = load_model('C:\\Users\\NIVEDHA\\Downloads\\final\\final\\image_classifier.h5')
disaster_model = load_model('C:\\Users\\NIVEDHA\\Downloads\\final\\final\\disaster_classifier.h5')

# Streamlit app
st.title("AI DISASTER MAPPING")

# File uploader for combined classification
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the selected image for combined classification
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict, caption="Uploaded Image for Classification", use_column_width=True)

    # Preprocess the image for both predictions
    image_to_predict = image_to_predict.resize((224, 224))
    img_array = image.img_to_array(image_to_predict)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make image damage prediction
    prediction_image = image_model.predict(img_array)
    result_image = "Damaged" if prediction_image[0][0] < 0.5 else "Undamaged"
    confidence_image = round(abs(prediction_image[0][0] - 0.5) * 200, 2)

    # Make disaster type prediction
    prediction_disaster = disaster_model.predict(img_array)
    result_disaster = "Flood" if prediction_disaster[0][0] < 0.5 else "Tornado"
    confidence_disaster = round(abs(prediction_disaster[0][0] - 0.5) * 200, 2)

    # Display results
    st.write(f"Image Damage Prediction: {result_image}")
    st.write(f"Damage Confidence: {confidence_image}%")
    st.write(f"Disaster Type Prediction: {result_disaster}")
    
