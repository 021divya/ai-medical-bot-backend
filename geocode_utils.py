from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="ai_medical_bot")

def geocode_location(location_text: str):
    """
    Converts user-entered location text into (lat, lng)
    with fallback handling
    """
    if not location_text or not location_text.strip():
        return None, None

    try:
        location = geolocator.geocode(
            location_text,
            addressdetails=True,
            timeout=10
        )

        if location:
            return location.latitude, location.longitude

    except Exception as e:
        print("Geocoding error:", e)

    return None, None
