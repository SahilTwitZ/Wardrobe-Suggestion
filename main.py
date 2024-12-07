import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a model for feature extraction
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Title of the app
st.title('Fashion Recommender System')


# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False


# Function for feature extraction from the uploaded image
def feature_extraction(img_path, model):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# Function to recommend similar images based on features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Extract features from uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Get recommendations based on extracted features
        indices = recommend(features, feature_list)

        # Display recommended images
        cols = st.columns(5)
        for i in range(5):
            cols[i].image(filenames[indices[0][i]])
    else:
        st.error("An error occurred during file upload")
