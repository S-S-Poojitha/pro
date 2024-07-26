import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Download TensorFlow Lite model
def download_model(url, model_path):
    response = requests.get(url)
    with open(model_path, 'wb') as file:
        file.write(response.content)

# Define model URL and path
model_url = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224.tgz'
model_path = 'mobilenet_v2.tflite'

# Download the model if not already present
if not os.path.exists(model_path):
    st.write("Downloading TensorFlow Lite model...")
    download_model(model_url, model_path)
    st.write("Model downloaded.")

# Load TensorFlow Lite model
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TensorFlow Lite model: {e}")
        return None

model = load_model()

# Function to extract features from an image using TensorFlow Lite model
def extract_features(image_path):
    if model is None:
        st.error("Model not loaded.")
        return None

    try:
        img = Image.open(image_path).resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.set_tensor(input_details[0]['index'], img)
        model.invoke()
        features = model.get_tensor(output_details[0]['index'])
        
        return features
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# Function to load the dataset and extract features for each image
def load_dataset_and_extract_features(dataset_path):
    image_paths = glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                  glob.glob(os.path.join(dataset_path, '*.jpeg'))
    feature_list = []
    for image_path in image_paths:
        features = extract_features(image_path)
        if features is not None:
            feature_list.append((image_path, features))
            print(f"Extracted features for image: {image_path}")  # Debug statement
    return feature_list

# Load dataset images and extract their features
dataset_images = load_dataset_and_extract_features('fashion_dataset')

# Function to find similar images based on extracted features
def find_similar_images(uploaded_image_features, dataset_images, top_n=3):
    if uploaded_image_features is None:
        st.error("Error: Uploaded image features are None.")
        return []

    similarities = []
    for image_path, features in dataset_images:
        try:
            similarity = cosine_similarity(uploaded_image_features, features)[0][0]
            similarities.append((image_path, similarity))
            print(f"Comparing with {image_path}, Similarity: {similarity}")  # Debug statement
        except Exception as e:
            st.error(f"Error during similarity calculation: {e}")
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit application
st.title('Product Image Search with TensorFlow Lite')

st.write("Upload an image to find similar images based on advanced features from the dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image_path = "uploaded_image.jpg"
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    uploaded_image.save(uploaded_image_path)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Extract features of the uploaded image
    uploaded_image_features = extract_features(uploaded_image_path)
    if uploaded_image_features is not None:
        st.write("Uploaded Image features extracted")

        # Find similar images
        similar_images = find_similar_images(uploaded_image_features, dataset_images)
        
        if similar_images:
            st.write("Similar Images:")
            for image_path, similarity in similar_images:
                st.image(image_path, caption=f"{os.path.basename(image_path)} (Similarity: {similarity:.2f})", use_column_width=True)
        else:
            st.write("No similar images found.")
    else:
        st.write("Failed to extract features from the uploaded image.")
