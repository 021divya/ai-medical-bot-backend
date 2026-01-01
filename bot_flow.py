from datetime import datetime
from predict_specialist import predict_specialist
from recommend_doctors import recommend_doctors

# =========================
# BOT GREETING
# =========================
def greet_user():
    hour = datetime.now().hour

    if hour < 12:
        greeting = "Good Morning ☀️"
    elif hour < 17:
        greeting = "Good Afternoon 🌤️"
    else:
        greeting = "Good Evening 🌙"

    return f"{greeting}\nI’m your medical assistant. How can I help you today?"


# =========================
# ASK SYMPTOMS
# =========================
def ask_symptoms():
    return "Please tell me your symptoms."


# =========================
# PROCESS SYMPTOMS
# =========================
def handle_symptoms(symptoms_text):
    specialist = predict_specialist(symptoms_text)

    response = (
        f"Based on your symptoms, you should consult a "
        f"{specialist} specialist.\n"
        "Now, please set your preferences."
    )

    return specialist, response


# =========================
# ASK FILTERS (UI SLIDERS)
# =========================
def ask_filters():
    """
    Frontend will show sliders for these.
    Bot only explains what is needed.
    """
    return {
        "message": (
            "Please adjust your preferences:\n"
            "- Maximum distance (3–5 km)\n"
            "- Maximum consultation fees\n"
            "- Minimum doctor rating"
        ),
        "expected_inputs": {
            "max_distance_km": [3, 4, 5],
            "max_fees": "number",
            "min_rating": [3.5, 4.0, 4.5]
        }
    }


# =========================
# FINAL RECOMMENDATION
# =========================
def get_recommendation(symptoms_text, user_filters):
    """
    user_filters comes from frontend sliders
    Example:
    {
        "max_distance_km": 5,
        "max_fees": 2000,
        "min_rating": 4.0
    }
    """

    results = recommend_doctors(symptoms_text)

    if results.empty:
        return "Sorry, no doctors match your preferences. Try expanding your filters."

    return {
        "message": "Here are the best doctors matching your preferences:",
        "doctors": results.to_dict(orient="records")
    }


# =========================
# TEST BOT FLOW (CLI)
# =========================
if __name__ == "__main__":
    print(greet_user())
    print()
    print(ask_symptoms())

    # Simulated user input
    user_symptoms = "chest pain and breathlessness"
    specialist, msg = handle_symptoms(user_symptoms)
    print("\nUSER:", user_symptoms)
    print("BOT:", msg)

    print("\nBOT:", ask_filters()["message"])

    # Simulated slider values
    filters = {
        "max_distance_km": 5,
        "max_fees": 2000,
        "min_rating": 4.0
    }

    final_response = get_recommendation(user_symptoms, filters)
    print("\nBOT:", final_response)
