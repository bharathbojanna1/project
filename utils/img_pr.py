import os
import numpy as np
import tensorflow as tf
import cv2
import json
import joblib
from skimage.feature import local_binary_pattern
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load Pretrained ResNet50 (without top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load class mapping
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_LABELS = {0: "Flea Allergy", 1: "Hotspot", 2: "Mange", 3: "Ringworm"}

# ---------------- Feature Extraction Functions ---------------- #

def extract_cnn_features(img):
    """Extract deep CNN features using ResNet50."""
    img = img.resize((224, 224))  # Resize for ResNet
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = base_model.predict(img_array)
    return features.flatten()  # Flatten to 1D feature vector

def extract_lbp_features(img):
    """Extract texture features using Local Binary Patterns (LBP)."""
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    # Extract LBP features
    lbp = local_binary_pattern(img_array, P=8, R=1, method="uniform")
    
    # Compute histogram of LBP values
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    
    return hist  # Return LBP histogram as feature vector


def load_svm_model():
    """Load the trained SVM model from disk."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

model_path = os.path.join(BASE_DIR, "..", "models", "svm_skin_disease2.pkl")


def predict_skin_disease(image):

    # Extract Features
    cnn_features = extract_cnn_features(image)
    lbp_features = extract_lbp_features(image)
    
    # Combine CNN + LBP Features
    final_features = np.hstack((cnn_features, lbp_features)).reshape(1, -1)

    # Load SVM Model & Predict
    svm_model = load_svm_model()
    prediction = svm_model.predict(final_features)
    confidence = max(svm_model.predict_proba(final_features)[0])

    return CLASS_LABELS.get(prediction[0], "Unknown"), confidence
