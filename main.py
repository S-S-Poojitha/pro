import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
from skimage.color import rgb2hsv
from skimage.feature import hog
from skimage import exposure

# Set the dataset path
dataset_path = 'fashion_dataset'

# Initialize the MobileNetV2 model without the top layer for feature extraction
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,  # Exclude the top layer to get feature vectors
        pooling='avg'      # Average pooling to get a feature vector
    )

model = load_model()

# Function to extract color features from an image
def extract_color_features(image):
    img = np.array(image)
    img_hsv = rgb2hsv(img)
    # Compute color histogram in HSV color space
    hist_h = np.histogram(img_hsv[:, :, 0], bins=32, range=(0, 1))[0]
    hist_s = np.histogram(img_hsv[:, :, 1], bins=32, range=(0, 1))[0]
    hist_v = np.histogram(img_hsv[:, :, 2], bins=32, range=(0, 1))[0]
    return np.concatenate([hist_h, hist_s, hist_v])

# Function to extract features from an image using MobileNetV2 and color features
def extract_features(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_rgb = np.array(img) / 255.0
    img_rgb = np.expand_dims(img_rgb, axis=0)
    features_mobilenet = model.predict(img_rgb).flatten()
    
    # Extract color features
    color_features = extract_color_features(img)
    
    # Combine MobileNetV2 features and color features
    combined_features = np.concatenate([features_mobilenet, color_features])
    return combined_features

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
        similarity = cosine_similarity(uploaded_image_features.reshape(1, -1), features.reshape(1, -1))[0][0]
        similarities.append((image_path, similarity))
        print(f"Comparing with {image_path}, Similarity: {similarity}")  # Debug statement
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit application
st.title('Product Image Search with TensorFlow and Color Features')

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
