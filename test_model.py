# test_ai_components.py

from dotenv import load_dotenv
load_dotenv()

import os
import torch
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
MODEL_A_PATH = './model_a/checkpoint-505'
MODEL_B_PATH = './model_b/checkpoint-404'

# --- 1. LOAD THE ENSEMBLE MODELS AND TOKENIZER (Unchanged) ---
print("Loading local ensemble models... This may take a moment.")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH)
    model_a = AutoModelForSequenceClassification.from_pretrained(MODEL_A_PATH)
    model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH)
    model_a.eval()
    model_b.eval()
    print("✅ Local models loaded successfully!")
except Exception as e:
    print(f"❌ FATAL: Could not load local models. Error: {e}")
    exit()

# --- 2. THE LOCAL PREDICTION FUNCTION (Unchanged) ---
def get_text_emotion(text: str):
    """Analyzes text using the local ensemble of two fine-tuned models."""
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
        print(f"Error during local model prediction: {e}")
        return "error", 0.0

# --- 3. THE *FINAL* OPENAI SUGGESTION FUNCTION ---
# <-- CHANGED: Now accepts the full journal_text
def get_openai_suggestion(journal_text: str, user_preferences: str):
    """Generates a highly personalized and context-aware suggestion."""
    try:
        if not os.environ.get('OPENAI_API_KEY'):
            return "SKIPPED: OPENAI_API_KEY not found in .env file."

        # <-- CHANGED: This is the new, more sophisticated prompt
        prompt = f"""
        A user has written the following journal entry:
        ---
        "{journal_text}"
        ---
        The user's interests are: {user_preferences}.

        Your Task:
        Write a short, empathetic, and human-like reply (2-3 sentences).
        1. Acknowledge the specifics of their entry (both the good and bad parts).
        2. Validate their feelings and their personal progress.
        3. If it feels natural, subtly suggest how ONE of their interests could relate to their situation. Do not just list their hobbies.
        """
            
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a wise and empathetic journaling assistant who replies to entries like a thoughtful friend."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80, # Increased slightly for more thoughtful replies
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return "ERROR: The API call failed. See console for details."


# --- 4. INTERACTIVE TESTING LOOP (Minor changes) ---
if __name__ == "__main__":
    print("\n--- 🚀 Interactive AI Component Tester (Context-Aware) ---")
    
    print("\nFirst, let's simulate a user's preferences.")
    preferences_input = input("Enter user's hobbies/interests > ")
    print("-" * 50)
    print("Great! Now you can start testing.")
    print("Type 'quit' or 'exit' to close.")
    print("-" * 50)


    while True:
        user_input = input("Enter text > ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting tester. Goodbye!")
            break
        
        if not user_input.strip():
            continue

        predicted_label, confidence_score = get_text_emotion(user_input)
        
        # <-- CHANGED: Pass the full user_input (journal_text) to the function
        ai_suggestion = get_openai_suggestion(user_input, preferences_input)

        print(f"  -> Predicted Emotion: {predicted_label.capitalize()}")
        print(f"  -> Confidence: {confidence_score:.2%}")
        print(f"  -> OpenAI Suggestion: {ai_suggestion}")
        print("-" * 25)