import requests
import streamlit as st
from components.config import earth911_api_key

BASE_URL = "http://api.earth911.com/"

@st.cache_data(show_spinner=False)
def get_material_id(specific_item):
    try:
        response = requests.get(f"{BASE_URL}/earth911.searchMaterials", params={
            "api_key": earth911_api_key,
            "query": specific_item
        })

        result = response.json().get("result", [])

        if result and "material_id" in result[0]:
            return result[0]["material_id"]
        else:
            st.warning("Please throw away trash through curbside pickup")
            return None

    except requests.exceptions.RequestException as e:
        st.error("Earth911 API request failed. Please report this on the About page.")
        st.exception(e)
        return None


@st.cache_data(show_spinner=False)
def get_postal_coordinates(zip_code):
    try:
        response = requests.get(f"{BASE_URL}/earth911.getPostalData", params={
            "api_key": earth911_api_key,
            "country": "US",
            "postal_code": zip_code
        })

        result = response.json().get("result")

        if result and "latitude" in result and "longitude" in result:
            return result["latitude"], result["longitude"]
        else:
            return None

    except requests.exceptions.RequestException as e:
        st.error("Earth911 API request failed. Please report this on the About page.")
        st.exception(e)
        return None


@st.cache_data(show_spinner=False)
def get_dropoff_locations(lat, lon, material_id):
    try:
        response = requests.get(f"{BASE_URL}/earth911.searchLocations", params={
            "api_key": earth911_api_key,
            "latitude": lat,
            "longitude": lon,
            "material_id": material_id,
            "max_distance": 20,
            "max_results": 5
        })

        result = response.json().get("result", [])
        if result:
            return result
        else:
            return None

    except requests.exceptions.RequestException as e:
        st.error("Earth911 API request failed. Please report this on the About page.")
        st.exception(e)
        return None


@st.cache_data(show_spinner=False)
def get_location_details(location_id):
    try:
        response = requests.get(f"{BASE_URL}/earth911.getLocationDetails", params={
            "api_key": earth911_api_key,
            "location_id": location_id
        })

        return response.json()["result"]

    except requests.exceptions.RequestException as e:
        st.error("Earth911 API request failed. Please report this on the About page.")
        st.exception(e)
        return None