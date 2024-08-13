import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Set the dataset path
dataset_path = 'fashion_dataset'

# Initialize the EfficientNetB0 model with include_top=True and classifier_activation='softmax'
@st.cache_resource
def load_model():
    return tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,  # Exclude the top layers for feature extraction
        pooling='avg'       # Use global average pooling instead of flattening
    )

model = load_model()

# Function to load and preprocess an image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0
    return img

# Function to extract features from a batch of images using EfficientNetB0
def extract_features_batch(image_paths, batch_size=32):
    images = np.array([preprocess_image(path) for path in image_paths])
    features = model.predict(images, batch_size=batch_size)
    return [(path, feature) for path, feature in zip(image_paths, features)]

# Function to load the dataset and extract features for each image in parallel with batch processing
@st.cache_resource
def load_dataset_and_extract_features(dataset_path):
    image_paths = glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                  glob.glob(os.path.join(dataset_path, '*.jpeg'))
    
    batch_size = 32
    feature_list = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_features = executor.submit(extract_features_batch, batch_paths, batch_size).result()
            feature_list.extend(batch_features)
    
    return feature_list

# Caching the similarity calculation to avoid recomputation
@lru_cache(maxsize=128)
def compute_similarity(uploaded_image_features, dataset_image_features):
    similarity = cosine_similarity(uploaded_image_features, dataset_image_features)[0][0]
    return similarity

# Function to find similar images based on extracted features using parallel processing
def find_similar_images(uploaded_image_features, dataset_images, top_n=3):
    similarities = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(compute_similarity, uploaded_image_features, features)
            for _, features in dataset_images
        ]
        
        for future, (image_path, features) in zip(futures, dataset_images):
            similarity = future.result()
            similarities.append((image_path, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit application
st.title('Product Image Search with TensorFlow & EfficientNetB0')

st.write("Upload an image to find similar images based on advanced features from the dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image_path = "uploaded_image.jpg"
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    uploaded_image.save(uploaded_image_path)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Extract features of the uploaded image
    uploaded_image_preprocessed = preprocess_image(uploaded_image_path)
    uploaded_image_features = model.predict(np.expand_dims(uploaded_image_preprocessed, axis=0))
    st.write("Uploaded Image features extracted")

    # Load dataset images and extract their features (with caching)
    dataset_images = load_dataset_and_extract_features(dataset_path)

    # Find similar images
    similar_images = find_similar_images(uploaded_image_features, dataset_images)
    
    if similar_images:
        st.write("Similar Images:")
        for image_path, similarity in similar_images:
            st.image(image_path, caption=f"{os.path.basename(image_path)} (Similarity: {similarity:.2f})", use_column_width=True)
    else:
        st.write("No similar images found.")
