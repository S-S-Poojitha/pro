import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import cv2
import requests
from bs4 import BeautifulSoup
import re

# Load pre-trained object detection model from TensorFlow Hub
model_url = 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1'  # Updated model URL
model = hub.load(model_url)

# Load pre-trained classification model from TensorFlow Hub
classifier_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'
classifier = hub.load(classifier_url)

# Function to perform object detection
def detect_objects(image):
    image_np = np.array(image)
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, [300, 300])
    image_tensor = image_tensor / 255.0  # Normalize
    image_tensor = image_tensor[tf.newaxis, ...]  # Add batch dimension

    results = model(image_tensor)
    boxes = results['detection_boxes'][0].numpy()
    scores = results['detection_scores'][0].numpy()
    classes = results['detection_classes'][0].numpy()

    return boxes, scores, classes

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, scores, threshold=0.5):
    image_np = np.array(image)
    for i, box in enumerate(boxes):
        if scores[i] > threshold:
            y1, x1, y2, x2 = box
            (startX, startY, endX, endY) = (int(x1 * image_np.shape[1]), int(y1 * image_np.shape[0]), int(x2 * image_np.shape[1]), int(y2 * image_np.shape[0]))
            cv2.rectangle(image_np, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return Image.fromarray(image_np)

# Function to classify the object
def classify_object(image):
    image_np = np.array(image)
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, [299, 299])
    image_tensor = image_tensor / 255.0  # Normalize
    image_tensor = image_tensor[tf.newaxis, ...]  # Add batch dimension

    features = classifier(image_tensor)
    return features.numpy()

# Function to search for images related to a query
def scrape_images(query):
    search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all("img")
    image_links = [img['src'] for img in img_tags if 'src' in img.attrs and img['src'].startswith('http')]
    return image_links

# Streamlit application
st.title('Enhanced Image Search with Object Detection')

st.write("Upload an image to detect objects and find related images based on the detected objects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Perform object detection
    boxes, scores, classes = detect_objects(uploaded_image)
    image_with_boxes = draw_boxes(uploaded_image, boxes, scores)
    st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

    # Example of handling detected objects
    if boxes is not None and len(boxes) > 0:
        st.write("Detected Objects:")
        for i, box in enumerate(boxes):
            if scores[i] > 0.5:
                st.write(f"Object {i + 1}: Class ID {int(classes[i])}, Score: {scores[i]:.2f}")
                
                # For simplicity, use class ID to query related images
                # In practice, you should use a more meaningful label
                class_label = f"object_class_{int(classes[i])}"
                st.write(f"Searching for images related to: {class_label}")

                # Scrape images related to the detected object
                image_links = scrape_images(class_label)
                
                st.write("Related Images Found:")
                for link in image_links:
                    st.image(link, caption='Related Image', use_column_width=True)
    else:
        st.write("No objects detected.")
