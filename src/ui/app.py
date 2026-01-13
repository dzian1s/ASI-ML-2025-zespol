import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.title("ML Demo – Prediction")

st.header("Input features")

neighbourhood_group = st.number_input(
    "neighbourhood_group", min_value=0, value=1, step=1
)
neighbourhood = st.number_input(
    "neighbourhood", min_value=0, value=45, step=1
)

latitude = st.number_input(
    "latitude", value=40720000
)
longitude = st.number_input(
    "longitude", value=-73990000
)

room_type = st.number_input(
    "room_type", min_value=0, value=2, step=1
)

minimum_nights = st.number_input(
    "minimum_nights", min_value=1, value=3, step=1
)
number_of_reviews = st.number_input(
    "number_of_reviews", min_value=0, value=128, step=1
)

reviews_per_month = st.number_input(
    "reviews_per_month", value=1.35
)

calculated_host_listings_count = st.number_input(
    "calculated_host_listings_count", min_value=0, value=2, step=1
)

availability_365 = st.number_input(
    "availability_365", min_value=0, max_value=365, value=210, step=1
)

if st.button("Predict"):
    payload = {
        "neighbourhood_group": int(neighbourhood_group),
        "neighbourhood": int(neighbourhood),
        "latitude": int(latitude),
        "longitude": int(longitude),
        "room_type": int(room_type),
        "minimum_nights": int(minimum_nights),
        "number_of_reviews": int(number_of_reviews),
        "reviews_per_month": float(reviews_per_month),
        "calculated_host_listings_count": int(calculated_host_listings_count),
        "availability_365": int(availability_365),
    }

    try:
        r = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15,
        )
        st.write("Status:", r.status_code)
        st.json(r.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
