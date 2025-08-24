# app/emotion.py

import io
import base64
import numpy as np
from PIL import Image
from fer import FER
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
# Replace with your actual Hugging Face username and repo name
HF_REPO_ID = "iamankn/manopriyam-emotion-ensemble" 

# --- GLOBAL MODEL CACHE ---
# We initialize the models as None. They will be loaded only when needed.
MODELS_CACHE = {
    "tokenizer": None,
    "model_a": None,
    "model_b": None
}

# --- Initialize face detector once ---
face_detector = FER(mtcnn=True)

# --- NEW LAZY LOADING FUNCTION ---
def load_models_if_needed():
    """Checks if models are loaded and loads them if they are not."""
    if MODELS_CACHE["tokenizer"] is None:
        print("LAZY LOADING: Models not found in cache. Loading from Hugging Face Hub...")
        try:
            MODELS_CACHE["tokenizer"] = AutoTokenizer.from_pretrained(HF_REPO_ID, subfolder="checkpoint-404")
            MODELS_CACHE["model_a"] = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID, subfolder="checkpoint-505")
            MODELS_CACHE["model_b"] = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID, subfolder="checkpoint-404")
            
            MODELS_CACHE["model_a"].eval()
            MODELS_CACHE["model_b"].eval()
            print("LAZY LOADING: Models loaded and cached successfully!")
        except Exception as e:
            print(f"FATAL: Could not lazy-load models. Error: {e}")

def get_text_emotion(text: str):
    """
    Analyzes text using an ensemble of two fine-tuned models.
    Loads models on the first run.
    """
    # Step 1: Ensure models are loaded before proceeding.
    load_models_if_needed()

    # Step 2: Check if loading failed.
    tokenizer = MODELS_CACHE["tokenizer"]
    model_a = MODELS_CACHE["model_a"]
    model_b = MODELS_CACHE["model_b"]

    if not all([tokenizer, model_a, model_b]):
        return "neutral", 0.0

    # Step 3: Perform prediction (same as before).
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits_a = model_a(**inputs).logits
            logits_b = model_b(**inputs).logits
        ensembled_logits = (logits_a + logits_b) / 2.0
        probabilities = torch.nn.functional.softmax(ensembled_logits, dim=-1)[0]
        best_prob_index = torch.argmax(probabilities).item()
        label = model_b.config.id2label[best_prob_index].lower()
        score = probabilities[best_prob_index].item()
        return label, score
    except Exception as e:
        print(f"Error during text emotion analysis: {e}")
        return "neutral", 0.0

def get_face_emotion(image_data_uri: str):
    """
    Decodes the base64 Data-URI, run FER, and returns (label: str, confidence: float).
    """
    try:
        header, b64 = image_data_uri.split(",", 1)
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img)
        top = face_detector.top_emotion(arr)
        if top is None:
            return None, 0.0
        label, score = top
        return label, float(score)
    except Exception as e:
        print(f"Error during face emotion analysis: {e}")
        return None, 0.0