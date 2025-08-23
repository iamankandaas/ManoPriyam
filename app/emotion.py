# app/emotion.py

import io
import base64
import numpy as np
from PIL import Image
from fer import FER
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
MODEL_A_PATH = './model_a/checkpoint-505'
MODEL_B_PATH = './model_b/checkpoint-404'

# Initialize face detector once
face_detector = FER(mtcnn=True)

# --- NEW ENSEMBLE MODEL LOADING ---
try:
    print("Loading ensemble models for text emotion...")
    # We only need one tokenizer as both models share the same kind
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH)
    
    model_a = AutoModelForSequenceClassification.from_pretrained(MODEL_A_PATH)
    model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH)
    
    # Put models in evaluation mode
    model_a.eval()
    model_b.eval()
    
    print("Ensemble models loaded successfully!")

except Exception as e:
    print(f"FATAL: Could not load the fine-tuned ensemble models. Error: {e}")
    # If the custom models can't load, we cannot proceed with text analysis.
    # We will create placeholder objects so the app doesn't crash on startup.
    tokenizer, model_a, model_b = None, None, None


def get_text_emotion(text: str):
    """
    Analyzes text using an ensemble of two fine-tuned models.
    Returns: (label: str, confidence: float)
    """
    # If models failed to load, return a safe default.
    if not all([tokenizer, model_a, model_b]):
        return "neutral", 0.0

    try:
        # 1. Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # 2. Get predictions (logits) from both models without computing gradients
        with torch.no_grad():
            logits_a = model_a(**inputs).logits
            logits_b = model_b(**inputs).logits

        # 3. Average the logits (the core of the ensemble)
        ensembled_logits = (logits_a + logits_b) / 2.0

        # 4. Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(ensembled_logits, dim=-1)[0]

        # 5. Get the winning label and its confidence score
        best_prob_index = torch.argmax(probabilities).item()
        
        # Use model_b's config to map the index back to a string label
        label = model_b.config.id2label[best_prob_index].lower()
        score = probabilities[best_prob_index].item()
        
        return label, score

    except Exception as e:
        print(f"Error during text emotion analysis: {e}")
        # If anything goes wrong during prediction, return a neutral fallback
        return "neutral", 0.0


def get_face_emotion(image_data_uri: str):
    """
    Decode the base64 Data-URI, run FER, 
    and return (label: str, confidence: float),
    or (None, 0.0) if no face is detected or on any error.
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
    except Exception:
        return None, 0.0