import os
import tensorflow as tf
import json
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Get API Key from environment
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Ensure API key is loaded
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ DeepSeek API key is missing. Please set it in .env file.")

# Define the model architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes: Ringworm, Mange, Hotspot, Flea Allergy
    ])
    return model

# Load class mapping
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "..", "models", "class_mapping.json")
with open(CLASS_MAPPING_PATH, "r") as f:
    CLASS_LABELS = json.load(f)
CLASS_LABELS = {int(k): v for k, v in CLASS_LABELS.items()}  # Convert keys to int

# Load the model
model_path = os.path.join(BASE_DIR, "..", "models", "SkinDisease.weights.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = build_model()
try:
    model.load_weights(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model weights: {str(e)}")

# Function to predict skin disease
def predict_skin_disease(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    
    return CLASS_LABELS.get(predicted_class, "Unknown"), float(tf.reduce_max(predictions))

# Function to use DeepSeek API
def ask_deepseek(question):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": "You are a veterinary AI expert."},
                     {"role": "user", "content": question}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ DeepSeek API error: {str(e)}"
