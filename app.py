import streamlit as st
from components.ui import (
    show_home_tab,
    show_locations_tab,
    show_how_to_use_tab,
    show_about_tab,
    show_news_tab,
    show_account_tab
)
from components.model import load_model, predict
from components.llm import generate_response, stream_response
from components.user_auth import login_screen, register_screen
import os

# Use polling to avoid file watcher issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

# Initialize session state defaults
for key, default in {
    "user_id": None,
    "is_logged_in": False,
    "screen": "login",
    "user_email": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Load Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# CSS Styling
st.markdown("""
<style>
#MainMenu { visibility:hidden; }

[data-testid="stAppViewContainer"] {
    background-image: radial-gradient(#444cf7 0.5px, #ffffff 0.5px);
    background-size: 10px 10px;
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
    background-image: radial-gradient(#444cf7 0.5px, rgba(255, 255, 255, 0.1) 0.5px);
    background-size: 10px 10px;
    z-index: 9999;
}
[data-testid="stHeaderLogo"] { opacity: 1 !important; }

iframe[title="streamlit_folium.st_folium"] { height: 300px; }

div.stButton > button:hover { transform: scale(1.015); transition: transform 0.2s ease; }

div.stButton > button { transition: transform 0.2s ease; }

[data-testid="stMetric"] {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border-radius: 12px !important;
}

[data-testid="stMetric"]:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.stToolbarActions { display: none !important; }

html, body { font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config("Green Bin", "assets/icon.png", layout="wide")
st.logo("assets/logo.png", size="large", icon_image="assets/icon.png")
st.image("assets/logo.png", width=200)

# Load model
model = load_model()

# Determine tabs dynamically
base_tabs = [
    ":material/home: Home",
    ":material/location_on: Locations",
    ":material/developer_guide: How to Use",
    ":material/news: Resources",
    ":material/info: About"
]

# Add Account tab only if logged in
if st.session_state.screen == "main" and st.session_state.user_id is not None:
    base_tabs.append(":material/account_circle: Account")

# Create tabs
tabs = st.tabs(base_tabs)

# Assign tab variables
tab1, tab2, tab3, tab4, tab5, *optional_tab = tabs

uploaded_file = None
picture = None

# Home tab
with tab1:
    if st.session_state.screen == "login":
        login_screen()
    elif st.session_state.screen == "register":
        register_screen()
    elif st.session_state.screen == "main":
        show_home_tab(model, predict, generate_response, stream_response)

# Locations tab
with tab2:
    show_locations_tab(uploaded_file, picture)

# How to Use tab
with tab3:
    show_how_to_use_tab()

# News tab
with tab4:
    show_news_tab()

# About tab
with tab5:
    show_about_tab()

# Account tab only if user is logged in
if optional_tab and st.session_state.user_id is not None:
    with optional_tab[0]:
        show_account_tab()