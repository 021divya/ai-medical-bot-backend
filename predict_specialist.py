import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Paths
# =========================
MODEL_DIR = "model"
LABEL_MAP_PATH = "model/label_map.json"

# =========================
# Load label map
# =========================
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

id2label = {int(v): k for k, v in label_map.items()}

# =========================
# Load tokenizer & model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# =========================
# Prediction function
# =========================
def predict_specialist(symptoms_text: str) -> str:
    text = symptoms_text.lower().strip()

    # -------------------------
    # General medicine first (vague symptoms)
    # -------------------------
    general_symptoms = [
        "nausea", "vomiting", "fever", "weakness",
        "fatigue", "dizziness", "cold", "cough", "flu"
    ]

    if any(word in text for word in general_symptoms):
        return "General Medicine"

    # -------------------------
    # Clear specialist cases
    # -------------------------
    if any(word in text for word in ["joint", "knee", "bone", "arthritis"]):
        return "Orthopedics"

    if any(word in text for word in ["skin", "rash", "itch", "acne"]):
        return "Dermatology"

    if any(word in text for word in ["chest", "heart", "palpitation"]):
        return "Cardiology"

    if any(word in text for word in ["headache", "migraine", "seizure"]):
        return "Neurology"

    # -------------------------
    # ML fallback
    # -------------------------
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

    return id2label.get(predicted_class, "General Medicine")







# =========================
# Test
# =========================
if __name__ == "__main__":
    test_text = "chest pain and breathlessness"
    result = predict_specialist(test_text)
    print("Predicted Specialist:", result)