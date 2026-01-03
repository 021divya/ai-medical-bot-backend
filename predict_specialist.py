import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# SAFE DEPLOY CONFIG
# =========================
MODEL_DIR = "model"

tokenizer = None
model = None

def load_ml_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        print("✅ ML model loaded")
    except Exception as e:
        tokenizer = None
        model = None
        print("⚠️ ML model not found, using rule-based only")

load_ml_model()

# =========================
# Prediction function
# =========================
def predict_specialist(symptoms_text: str) -> str:
    text = symptoms_text.lower().strip()

    # -------------------------
    # RULE-BASED (PRIMARY)
    # -------------------------
    if any(w in text for w in ["joint", "knee", "bone", "arthritis"]):
        return "Orthopedics"

    if any(w in text for w in ["skin", "rash", "itch", "acne"]):
        return "Dermatology"

    if any(w in text for w in ["chest", "heart", "palpitation"]):
        return "Cardiology"

    if any(w in text for w in ["headache", "migraine", "seizure"]):
        return "Neurology"

    if any(w in text for w in ["fever", "vomiting", "cold", "weakness", "fatigue"]):
        return "General Medicine"

    # -------------------------
    # ML FALLBACK (OPTIONAL)
    # -------------------------
    if tokenizer and model:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return str(predicted_class)

    # -------------------------
    # SAFE DEFAULT
    # -------------------------
    return "General Medicine"


# =========================
# Local Test
# =========================
if __name__ == "__main__":
    print(predict_specialist("chest pain and breathlessness"))
