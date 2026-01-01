import pandas as pd
from predict_specialist import predict_specialist
from distance_utils import get_distance_km

# =========================
# Load cleaned dataset
# =========================
doctor_df = pd.read_csv("data/clean_doctor_dataset.csv")

# Normalize column names
doctor_df.columns = (
    doctor_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Normalize text columns once
doctor_df["area"] = doctor_df["area"].astype(str).str.lower().str.strip()
doctor_df["speciality"] = doctor_df["speciality"].astype(str).str.lower().str.strip()

# =========================
# Recommendation Engine
# =========================
def recommend_doctors(
    symptoms_text: str,
    patient_lat: float,
    patient_lng: float,
    location_text: str,
    max_distance_km: int,
    max_fees: int,
    min_rating: float
):
    # -------------------------------------------------
    # 1️⃣ Predict specialist (robust)
    # -------------------------------------------------
    specialist = predict_specialist(symptoms_text).lower()

    SPECIALITY_MAP = {
        "orthopedics": ["orthopedics", "orthopaedics", "ortho"],
        "cardiology": ["cardiology", "cardiologist"],
        "dermatology": ["dermatology", "dermatologist"],
        "neurology": ["neurology", "neurologist"],
        "general medicine": ["general medicine", "physician", "general"]
    }

    allowed_specialities = SPECIALITY_MAP.get(specialist, [specialist])

    df_specialist = doctor_df[
        doctor_df["speciality"].isin(allowed_specialities)
    ].copy()

    if df_specialist.empty:
        return pd.DataFrame()

    # -------------------------------------------------
    # 2️⃣ STRICT LOCALITY FILTER (USER EXPECTATION 🔥)
    # -------------------------------------------------
    user_area = location_text.lower().split(",")[0].strip()

    locality_df = df_specialist[
        df_specialist["area"].str.startswith(user_area)
    ].copy()

    # 👉 If locality match exists, use ONLY that
    if not locality_df.empty:
        df_base = locality_df
        locality_used = True
    else:
        # fallback to all specialists (distance-based)
        df_base = df_specialist
        locality_used = False

    # -------------------------------------------------
    # 3️⃣ Distance calculation (Haversine)
    # -------------------------------------------------
    df_base["distance_km"] = df_base.apply(
        lambda row: get_distance_km(
            patient_lat,
            patient_lng,
            row["latitude"],
            row["longitude"]
        ),
        axis=1
    )

    # -------------------------------------------------
    # 4️⃣ Distance-based filtering (auto-expand)
    # -------------------------------------------------
    DISTANCE_LEVELS = [3, 5, 10]

    for radius in DISTANCE_LEVELS:
        if radius < max_distance_km:
            continue

        df = df_base[
            (df_base["distance_km"] <= radius) &
            (df_base["fees"] <= max_fees) &
            (df_base["rating"] >= min_rating)
        ].copy()

        if not df.empty:
            df["used_radius_km"] = radius
            df["match_type"] = "locality" if locality_used else "distance"
            return df.sort_values(
                by=["rating", "distance_km"],
                ascending=[False, True]
            )

    # -------------------------------------------------
    # 5️⃣ Nothing found
    # -------------------------------------------------
    return pd.DataFrame()
