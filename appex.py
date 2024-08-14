import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import cv2  # For color similarity using histogram comparison

# Set the dataset path
dataset_path = 'fashion_dataset'

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=True,
        classifier_activation='softmax'
    )

model = load_model()

def extract_features(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features

def assign_label(uploaded_image_features):
    predictions = model.predict(uploaded_image_features)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0][1]  # Return the label name

def load_images_from_directory(directory):
    image_paths = glob.glob(os.path.join(directory, '*.jpg')) + \
                  glob.glob(os.path.join(directory, '*.jpeg')) + \
                  glob.glob(os.path.join(directory, '*.png'))
    return image_paths

def compute_color_histogram(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return normalize(hist, norm='l2')

def find_similar_images(uploaded_image_features, color_histogram, directory, top_n=3):
    image_paths = load_images_from_directory(directory)
    similarities = []
    
    for image_path in image_paths:
        features = extract_features(image_path)
        shape_similarity = cosine_similarity(uploaded_image_features, features)[0][0]
        
        hist = compute_color_histogram(image_path)
        color_similarity = cv2.compareHist(color_histogram, hist, cv2.HISTCMP_CORREL)
        
        combined_similarity = shape_similarity * 0.5 + color_similarity * 0.5  # Weighted similarity
        similarities.append((image_path, combined_similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

st.title('Product Image Search with TensorFlow')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image_path = "uploaded_image.jpg"
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    uploaded_image.save(uploaded_image_path)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Extract features of the uploaded image
    uploaded_image_features = extract_features(uploaded_image_path)
    
    # Assign a label to the uploaded image
    label = assign_label(uploaded_image_features)
    st.write(f"Assigned Label: {label}")
    
    # Navigate to the corresponding directory
    target_directory = os.path.join(dataset_path, label.replace(' ', '_').lower())
    
    if os.path.exists(target_directory):
        # Compute the color histogram for the uploaded image
        uploaded_histogram = compute_color_histogram(uploaded_image_path)
        
        # Find similar images in the target directory
        similar_images = find_similar_images(uploaded_image_features, uploaded_histogram, target_directory)
        
        if similar_images:
            st.write("Similar Images:")
            for image_path, similarity in similar_images:
                st.image(image_path, caption=f"{os.path.basename(image_path)} (Similarity: {similarity:.2f})", use_column_width=True)
        else:
            st.write("No similar images found.")
    else:
        st.write(f"No directory found for the assigned label: {label}")
