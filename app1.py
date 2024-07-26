import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity

# Set the dataset path
dataset_path = 'fashion_dataset'

# Initialize the MobileNetV2 model with include_top=True and classifier_activation='softmax'
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=True,  # Set include_top to True to include the fully-connected layer
        classifier_activation='softmax'  # Use softmax activation for classification
    )

model = load_model()

# Function to extract features from an image using MobileNetV2
def extract_features(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features

# Function to load the dataset and extract features for each image
@st.cache_resource
def load_dataset_and_extract_features(dataset_path):
    image_paths = glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                  glob.glob(os.path.join(dataset_path, '*.jpeg'))
    feature_list = []
    for image_path in image_paths:
        features = extract_features(image_path)
        feature_list.append((image_path, features))
        print(f"Extracted features for image: {image_path}")  # Debug statement
    return feature_list

# Load dataset images and extract their features
dataset_images = load_dataset_and_extract_features(dataset_path)

# Function to find similar images based on extracted features
def find_similar_images(uploaded_image_features, dataset_images, top_n=3):
    similarities = []
    for image_path, features in dataset_images:
        similarity = cosine_similarity(uploaded_image_features, features)[0][0]
        similarities.append((image_path, similarity))
        print(f"Comparing with {image_path}, Similarity: {similarity}")  # Debug statement
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit application
st.title('Product Image Search with TensorFlow')

st.write("Upload an image to find similar images based on advanced features from the dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image_path = "uploaded_image.jpg"
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    uploaded_image.save(uploaded_image_path)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Extract features of the uploaded image
    uploaded_image_features = extract_features(uploaded_image_path)
    st.write("Uploaded Image features extracted")

    # Find similar images
    similar_images = find_similar_images(uploaded_image_features, dataset_images)
    
    if similar_images:
        st.write("Similar Images:")
        for image_path, similarity in similar_images:
            st.image(image_path, caption=f"{os.path.basename(image_path)} (Similarity: {similarity:.2f})", use_column_width=True)
    else:
        st.write("No similar images found.")
