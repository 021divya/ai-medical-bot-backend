from fastapi import FastAPI
from pydantic import BaseModel
from bot_flow import greet_user, handle_symptoms
from recommend_doctors import recommend_doctors
from geocode_utils import geocode_location

app = FastAPI(
    title="AI Medical Assistant Bot",
    description="Symptom-based doctor recommendation system",
    version="1.0"
)

# =========================
# Request Models
# =========================

class SymptomRequest(BaseModel):
    symptoms: str


class FilterRequest(BaseModel):
    symptoms: str
    location_text: str      # User enters location as text (Ola-style)
    max_distance_km: int    # 3 / 5 / etc.
    max_fees: int
    min_rating: float


# =========================
# API Endpoints
# =========================

@app.get("/greet")
def greet():
    """
    Bot greeting based on time
    """
    return {
        "message": f"{greet_user()} 👋 I’m your AI medical assistant. Please tell me what symptoms you are experiencing.",
        "next_actions": ["enter_symptoms"]
    }


@app.post("/symptoms")
def process_symptoms(data: SymptomRequest):
    """
    Takes symptoms and predicts specialist
    """
    specialist, response = handle_symptoms(data.symptoms)

    return {
        "specialist": specialist,
        "message": f"{response} Please enter your location so I can find nearby doctors.",
        "next_actions": ["enter_location"]
    }


@app.post("/recommend")
def recommend(data: FilterRequest):
    """
    Recommends doctors based on:
    - symptoms
    - location text
    - distance (auto-expand)
    - fees
    - rating
    """

    # 1️⃣ Convert location text → latitude & longitude
    lat, lng = geocode_location(data.location_text)

    if lat is None or lng is None:
        return {
            "message": "I couldn’t understand the location you entered. Please try entering a nearby area or locality.",
            "doctors": [],
            "next_actions": ["reenter_location"]
        }

    # 2️⃣ Call recommendation engine
    results = recommend_doctors(
        symptoms_text=data.symptoms,
        patient_lat=lat,
        patient_lng=lng,
        location_text=data.location_text,
        max_distance_km=data.max_distance_km,
        max_fees=data.max_fees,
        min_rating=data.min_rating
    )

    if results.empty:
        return {
            "message": (
                "I couldn’t find doctors matching your preferences nearby. "
                "You may try increasing the distance or adjusting filters."
            ),
            "doctors": [],
            "next_actions": ["change_filters", "search_another_symptom"]
        }

    # 3️⃣ Read auto-expanded radius (if applied)
    used_radius = None
    if "used_radius_km" in results.columns:
        used_radius = int(results["used_radius_km"].iloc[0])

    # 4️⃣ Prepare response doctors list
    doctors = results[
        [
            "doctor_name",
            "area",
            "distance_km",
            "rating",
            "fees",
            "contact",
            "address",
            "availability_text"
        ]
    ].to_dict(orient="records")

    # 5️⃣ Dynamic, user-friendly message
    if used_radius:
        message = (
            f"Here are the best doctors found within {used_radius} km of "
            f"{data.location_text}, based on your preferences."
        )
    else:
        message = "Here are the best doctors near you based on your preferences."

    return {
        "message": message,
        "filters_applied": {
            "location": data.location_text,
            "max_distance_km": used_radius,
            "max_fees": data.max_fees,
            "min_rating": data.min_rating
        },
        "doctors": doctors,
        "next_actions": [
            "change_filters",
            "search_another_symptom"
        ]
    }


@app.post("/reset")
def reset():
    """
    Resets the conversation flow
    """
    return {
        "message": "Alright 😊 Let’s start fresh. Please tell me what symptoms you are experiencing.",
        "next_actions": ["enter_symptoms"]
    }
