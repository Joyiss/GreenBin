import keras
import numpy as np
from huggingface_hub import hf_hub_download
import streamlit as st
from components.config import class_names
from io import BytesIO

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="AIforGreat/TrashClassification", filename="efficientnet_garbage_classifier.keras")
    return keras.models.load_model(model_path)

def predict(model, file):
    # Convert uploaded file to a PIL image
    img = keras.preprocessing.image.load_img(BytesIO(file.read()), target_size=(224, 224))

    # Convert to array and preprocess
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.efficientnet.preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index]) * 100
    return class_names[index], confidence
