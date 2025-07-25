import streamlit as st
import keras
import cv2
import numpy as np
import google.generativeai as genai
import time
import requests
import folium
from streamlit_folium import st_folium
import random
import hashlib
from supabase import create_client, Client
import re

gemini_api_key = st.secrets["GEMINI_API_KEY"]
base_url = 'http://api.earth911.com/'
earth911_api_key = st.secrets["EARTH911_API_KEY"]
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]

# Setting up supabase for the backend
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(url, key)

# Setting up Gemini Model
genai.configure(api_key=gemini_api_key)

gen_model = genai.GenerativeModel("gemini-1.5-flash")

class_names = [
    "Battery",
    "Biological",
    "Brown-glass",
    "Cardboard",
    "Clothes",
    "Green-glass",
    "Metal",
    "Paper",
    "Plastic",
    "Shoes",
    "Trash",
    "White-glass"
]

tips = {
    "Battery": [
        "**Tip:** Be sure to safely wrap the batteries before disposing",
        "**Tip:** Store the batteries in a cool, dry place",
        "**Tip:** Check for any signs of bulging or damage before disposing",
        "**Tip:** Drop the batteries to recycle within six months, ensuring they are bagged or taped"
    ],
    "Biological": [
        "**Tip:** Compost food scraps and yard waste when possible",
        "**Tip:** Never mix biological waste with recyclables",
        "**Tip:** Use sealed bins to prevent odor and pests"
    ],
    "Brown-glass": [
        "**Tip:** Rinse glass bottles before recycling",
        "**Tip:** Remove any caps or lids",
        "**Tip:** Only recycle whole bottles — broken glass may not be accepted"
    ],
    "Cardboard": [
        "**Tip:** Flatten cardboard boxes to save space",
        "**Tip:** Remove excess tape or labels",
        "**Tip:** Do not recycle wax-coated or greasy cardboard (e.g. pizza boxes)"
    ],
    "Clothes": [
        "**Tip:** Donate gently used clothing to charity or thrift stores",
        "**Tip:** Recycle worn-out clothes through textile recycling programs",
        "**Tip:** Do not place clothing in curbside bins unless your area accepts it"
    ],
    "Green-glass": [
        "**Tip:** Rinse bottles to remove residue",
        "**Tip:** Remove metal or plastic lids before recycling",
        "**Tip:** Recycle only whole glass bottles, not shattered pieces"
    ],
    "Metal": [
        "**Tip:** Rinse food and drink cans before recycling",
        "**Tip:** Leave labels on — most facilities can remove them",
        "**Tip:** Avoid recycling sharp or rusted metal in curbside bins"
    ],
    "Paper": [
        "**Tip:** Recycle clean and dry paper only",
        "**Tip:** Do not recycle paper with food stains, grease, or water damage",
        "**Tip:** Staples and paper clips are okay — no need to remove them"
    ],
    "Plastic": [
        "**Tip:** Rinse plastic containers before placing them in the bin",
        "**Tip:** Check for recycling symbols #1 or #2 — most accepted curbside",
        "**Tip:** Leave caps on unless otherwise instructed"
    ],
    "Shoes": [
        "**Tip:** Donate usable shoes to shelters or reuse programs",
        "**Tip:** Recycle worn-out shoes through brand take-back programs",
        "**Tip:** Do not throw shoes in curbside recycling unless accepted"
    ],
    "Trash": [
        "**Tip:** Place dirty, contaminated, or non-recyclable items in the trash",
        "**Tip:** Avoid putting electronics, batteries, or hazardous waste in the trash",
        "**Tip:** Try to reduce trash by reusing or composting when possible"
    ],
    "White-glass": [
        "**Tip:** Rinse glass containers before recycling",
        "**Tip:** Remove any plastic or metal lids",
        "**Tip:** Recycle only whole glass bottles, not broken pieces"
    ]
}


# Loading the classification model
@st.cache_resource
def load_model():
    return keras.models.load_model('model/trashClassifier.keras')

model = load_model()

# Image preprocessing / Making Predictions
def predict(file):
    img_np = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    prediction = model.predict(np.array([img]) / 255.0)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index]) * 100
    return class_names[index], confidence

# Generating the response from the Gemini LLM
def generate_response(prediction, confidence):
    prompt = f"""
    You are a smart waste disposal assistant that helps users with their trash. You are going to get a prediction
    from a CNN Model on what the object is and you have to analyze the following object and provide a clear, friendly response that includes: 

    The classification: **Is this recyclable, compostable, or trash?** (Say only one — don't mention what it is *not*)  
    Briefly explain why it fits in that category only if it is not trash. Focus only on why it belongs in that category — do not explain why it isn’t in the others.
    A fun fact about the item (add an emoji if appropriate)  
    If confidence is below 90%, let the user know that the the classification may be inaccurate  
    A reminder: 📍 *To find where to dispose of this item, go to the Locations tab.*

    If the object name is too broad, generalize it to the most common example:
    - **Metal:** aluminum cans, steel cans  
    - **Biological:** food scraps, leaves, fruits, rotten vegetables, moldy bread  
    - **Trash:** dirty diapers, face masks, toothbrushes  

    Do not tell the user to check with their local recycling center — that warning has already been provided.
    
    **Use first person POV for user engagement even if you are talking about the CNN Model**

    Here is the object: **{prediction}**  
    Here is the confidence score: **{confidence:.1f}%**
    """

    response = gen_model.generate_content(prompt)
    return response

# Real-time generation effect
def stream_response(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.08)

# Get material ID for the specific item using Earth911 API
@st.cache_data(show_spinner=False)
def get_material_id(specific_item, api_key, base_url):
    try:
        response = requests.get(base_url + "/earth911.searchMaterials", params={
            "api_key": api_key,
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

# Get latitude, longitude coordinates from zip code using Earth911 API
@st.cache_data(show_spinner=False)
def get_postal_coordinates(zip_code, api_key, base_url):
    try:
        response = requests.get(base_url + "/earth911.getPostalData", params={
            "api_key": api_key,
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

# Search for drop-off centers for the item near user's location using Earth911 API
@st.cache_data(show_spinner=False)
def get_dropoff_locations(lat, lon, material_id, api_key, base_url):
    try:
        response = requests.get(base_url + "/earth911.searchLocations", params={
            "api_key": api_key,
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


# Get the drop-off centers' location details to display to the users
@st.cache_data(show_spinner=False)
def get_location_details(api_key, id):
    try:
        response = requests.get(base_url + "/earth911.getLocationDetails", params={
            "api_key": api_key,
            "location_id": id
        })

        return response.json()["result"]

    except requests.exceptions.RequestException as e:
        st.error("Earth911 API request failed. Please report this on the About page.")
        st.exception(e)
        return None

# Generate a unique hash based on the bytes
def get_hash(bytes):
    hasher = hashlib.sha256()
    hasher.update(bytes)
    return hasher.hexdigest()

# Uploads the image to supabase if the image does not already exist
def upload_misclassified_image(image, true_class, mime_type):
    image_bytes = image.read()
    new_hash = get_hash(image_bytes)
    new_filename = f"{new_hash}.jpg"
    path = f"Tmisclassified-images/{true_class}/{new_filename}"

    folders = supabase.storage.from_("misclassified-images").list("Tmisclassified-images")

    #Check for duplicate across all folders
    for folder in folders:
        folder_name = folder['name'].strip("/")
        files = supabase.storage.from_("misclassified-images").list(f"Tmisclassified-images/{folder_name}/")
        for file in files:
            if file["name"] == new_filename:
                st.warning("Image already uploaded")
                return

    supabase.storage.from_("misclassified-images").upload(path, image_bytes, {"content-type": mime_type})

# Basic settings for the website
st.set_page_config("Green Bin", "Images/icon.png", layout="wide")
st.logo("Images/logo.png", size="large", icon_image="Images/icon.png")
st.image("Images/logo.png", width=200)

# Creating the separate tabs for each section
tab1, tab2, tab3, tab4 = st.tabs([":material/home: Home", ":material/location_on: Locations", ":material/developer_guide: How to Use", ":material/info: About"])

# Hides the Main Menu using CSS
st.markdown(
    """
    <style>
    #MainMenu {
        visibility:hidden;
    }
""", unsafe_allow_html=True)

# Sets a bg pattern using CSS and sets the height for the folium map
st.markdown( """
<style>
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

[data-testid="stHeaderLogo"] {
    opacity: 1 !important;
}

iframe[title="streamlit_folium.st_folium"] {
    height: 300px;
}
</style>
""", unsafe_allow_html=True)

# Sets the apps font using Google Fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# Adds a hover effect on the buttons
st.markdown(
    """
    <style>
    div.stButton > button:hover {
        transform: scale(1.015);
        transition: transform 0.2s ease;
    }
    div.stButton > button {
        transition: transform 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hides the toolbar actions from the user
st.markdown(
    """
    <style>
    .stToolbarActions {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with tab1:
    col1, col2 = st.columns(2)

    # Allows users to upload/take a picture of the item
    with col1:
        uploaded_file = st.file_uploader("Please select a file", type="jpg")
        st.divider()
        enable = st.toggle("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)
        predict_button = st.button("Analyze :brain:", use_container_width=True)


    with col2:
        if predict_button:
            if uploaded_file and picture:
                st.warning("Please only provide one image")
            elif uploaded_file:
                st.image(uploaded_file, width=300)
                uploaded_file.seek(0)
                with st.spinner("Sorting Trash..."):
                    # Uses the predict function to get the prediction, and conf score of the model
                    model_prediction, confidence = predict(uploaded_file)
                    # Generates the response form the LLM
                    gen_model_text = generate_response(model_prediction, confidence)
                    st.write(f"**Confidence: {confidence:.2f}%**")
                    # Uses the write_stream function to create a real-time generation effect
                    st.write_stream(stream_response(gen_model_text.text))
                    st.session_state["model_prediction"] = model_prediction
                item_tips = tips.get(model_prediction)
                # Displays a random tip
                st.info(random.choice(item_tips), icon="💡")
                st.balloons()
            elif picture:
                st.image(picture, width=300)
                with st.spinner("Sorting Trash..."):
                    model_prediction, confidence = predict(picture)
                    gen_model_text = generate_response(model_prediction, confidence)
                    st.write(f"**Confidence: {confidence:.2f}%**")
                    st.write_stream(stream_response(gen_model_text.text))
                    st.session_state["model_prediction"] = model_prediction
                item_tips = tips.get(model_prediction)
                st.toast(random.choice(item_tips), icon="💡")
                st.balloons()
            else:
                st.warning("Please provide an image")


with tab2:
    st.header("Drop Off Locations :package:")

    tab4_col1, tab4_col2 = st.columns(2)

    if "zip_code" not in st.session_state:
        st.session_state["zip_code"] = ""
    if "prediction_correct" not in st.session_state:
        st.session_state["prediction_correct"] = ""
    if "user_select" not in st.session_state:
        st.session_state["user_select"] = ""
    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False
    if "allow_images" not in st.session_state:
        st.session_state["allow_images"] = False

    # All the items in Earth911 database
    specific_items = {
        "Battery": ["Alkaline Batteries", "Button Cell Batteries", "Car Batteries", "Lead-acid Batteries",
                    "Lithium Batteries", "Lithium-ion Batteries", "Marine Batteries", "Nickel-cadmium Batteries",
                    "Nickel-metal Hydride Batteries", "Nickel-zinc Batteries", "Silver-oxide Batteries",
                    "Zinc-air Batteries", "Zinc-carbon Batteries"],
        "Plastic": ["#1 Plastic Bags", "#1 Plastic Trays", "#2 Plastic Clamshells", "#3 Plastic Bags", "#4 Plastic Bags",
                    "#5 Plastic Bottles", "#5 Rigid Plastics", "#6 Plastic Cups", "#7 Plastic Bags", "#1 Plastic Beverage Bottles",
                    "#1 Rigid Plastics", "#2 Plastic Film", "#3 Plastic Bottles", "#4 Plastic Bottles", "#5 Plastic Caps",
                    "#6 Plastic - Expanded", "#6 Plastic Cups - Expanded", "#7 Plastic Bottles", "#1 Plastic Clamshells", "#2 Plastic Bags",
                    "#2 Plastic Jugs - Clear", "#3 Plastic Film", "#4 Plastic Film", "#5 Plastic Clamshells", "#6 Plastic Bags",
                    "#6 Plastic Film", "#7 Plastic Film", "#1 Plastic Film", "#2 Plastic Bottles", "#2 Plastic Jugs - Colored",
                    "#3 Rigid Plastics", "#4 Rigid Plastics", "#5 Plastic Cups", "#6 Plastic Bottles", "#6 Plastic Peanuts",
                    "#7 Rigid Plastics", "#1 Plastic Non-Beverage Bottles", "#2 Plastic Caps", "#2 Rigid Plastics", "#4 Flexible Plastics",
                    "#5 Plastic Bags", "#5 Plastic Film", "#6 Plastic Clamshells", "#6 Rigid Plastics", "Acrylics"],
        "Brown-glass": ["Brown Glass Beverage Containers", "Brown Glass Containers"],
        "Green-glass": ["Green Glass Beverage Containers", "Green Glass Containers"],
        "White-glass": ["Clear Glass Beverage Containers", "Clear Glass Containers"],
        "Clothes": ["Clothing"],
        "Shoes": ["Shoes"],
        "Metal": ["Aerosol Cans - Full", "Aluminum Trays", "Refrigerators", "Aluminum Beverage Cans", "Ferrous Metals",
                  "Steel Cans", "Aluminum Foil", "Metal Paint Cans", "Steel Lids", "Aluminum Food Cans",
                  "Metal Tags", "Washer/Dryers", "Aluminum Pie Plates", "Nonferrous Metals"],
        "Cardboard": ["Cardboard"],
        "Paper": ["Corrugated Cardboard", "Multi-wall Paper Bags", "Paper Sleeves", "Drink Boxes", "Newspaper",
                  "Paperback Books", "Envelopes", "Office Paper", "Paperboard", "Magazines",
                  "Paper Cups", "Phone Books", "Mixed Paper", "Paper Labels", "Wet-strength Paperboard"],
        "Biological": ["Organic Food Waste"],
        "Trash": ["Trash"]
    }

    with tab4_col1:
        if "model_prediction" in st.session_state:
            # Obtaining the information from the user
            st.session_state.zip_code = st.text_input("Enter your ZIP Code")
            st.session_state.prediction_correct = st.radio("Was the prediction correct?", ("Yes", "No"))

            # If the prediction was incorrect, users can choose to allow future training with their images
            if st.session_state.prediction_correct == "No":
                st.session_state.allow_images = st.checkbox("Allow training with my images",
                                                            help="By enabling this, your image may help the AI get smarter over time.")
                st.session_state.user_select = st.selectbox("What was the object?", class_names)
            else:
                st.session_state.user_select = st.session_state.model_prediction

            # Users can choose a specific item in the category for the best results
            specific_item = st.selectbox(
                f"What type of {st.session_state.user_select.lower()}?",
                specific_items[st.session_state.user_select], help="Choose what type of item"
            )

            if st.button("See Locations", use_container_width=True):
                if len(st.session_state.zip_code) == 5 and st.session_state.zip_code.isdigit():
                    coordinates = get_postal_coordinates(st.session_state.zip_code, earth911_api_key, base_url)
                    if coordinates is None:
                        st.warning("ZIP code not found. Please enter a valid U.S. ZIP code.")
                    else:
                        with st.spinner("Searching for locations..."):
                            # Checks if the prediction was incorrect and stores the image in supabase using upload_misclassified_image function
                            if st.session_state.prediction_correct == "No":
                                if st.session_state.allow_images:
                                    if uploaded_file:
                                        mime_type = uploaded_file.type
                                        upload_misclassified_image(uploaded_file, st.session_state.user_select.lower(),
                                                                   mime_type)
                                    else:
                                        mime_type = picture.type
                                        upload_misclassified_image(picture, st.session_state.user_select.lower(), mime_type)

                            lat, lon = coordinates
                            st.session_state.submitted = True

                            # Gets the material id for the item
                            material_id = get_material_id(specific_item, earth911_api_key, base_url)
                            if material_id is not None:
                                # Gets the drop-off centers location details
                                locations = get_dropoff_locations(lat, lon, material_id, earth911_api_key, base_url)

                                # Stores the information for each location in list
                                if locations is not None:
                                    st.session_state["coordinates"] = [
                                        {"latitude": float(loc["latitude"]), "longitude": float(loc["longitude"]),
                                         "description": loc["description"], "location_id": loc["location_id"]}
                                        for loc in locations
                                    ]

                                    # Sets the Folium Map
                                    first_coord = st.session_state["coordinates"][0]
                                    map = folium.Map(location=[first_coord["latitude"], first_coord["longitude"]],
                                                     zoom_start=8)

                                    ids = []

                                    # Adds all the locations and details on the map
                                    for loc in st.session_state["coordinates"]:
                                        marker = loc["latitude"], loc["longitude"]
                                        ids.append(loc["location_id"])
                                        folium.Marker(
                                            location=marker,
                                            popup=loc["description"],
                                            tooltip=loc["description"],
                                            icon=folium.Icon(icon="recycle", prefix="fa", color="blue")
                                        ).add_to(map)

                                    with tab4_col2:
                                        with st.container():
                                            # Displays the Map
                                            st_folium(map, use_container_width=True, returned_objects=[])

                                            # Gets the information of the drop-off locations to display to the user
                                            for location_id in ids:
                                                result = get_location_details(api_key=earth911_api_key, id=location_id)
                                                address = result[location_id]["address"]
                                                name = result[location_id]["description"]
                                                url = result[location_id]["url"]
                                                phone = result[location_id]["phone"]
                                                hours = result[location_id]["hours"]

                                                # Displays the information in an expander for each location
                                                with st.expander(name):
                                                    st.write(f"**Address**: {address}")
                                                    st.write(f"**Hours**: {hours}")
                                                    st.write(f"**Phone**: {phone}")
                                                    st.write(f"**Website**: [{url}]({url})")
                                else:
                                    st.warning("No nearby locations accept this item.")
                else:
                    st.warning("Please enter a valid 5-digit ZIP code.")
        else:
            st.warning("Please upload an image on the Home page")

with tab3:
    st.header(":recycle: How to Use", anchor=False)

    tab3_col1,tab3_col2 = st.columns(2)

    steps = {
        "Upload": "Take a photo of your item.",
        "Advice": "See if it's recyclable, compostable, or trash.",
        "Find": "Locate nearby recycling centers.",
        "Dispose": "Reduce waste responsibly!",
    }

    # Displaying the steps using basic CSS for clean UI
    for i, (title, desc) in enumerate(steps.items(), 1):
        if i%2 != 0:
            with tab3_col1:
                st.markdown(f"""
                    <div style='
                        background-color:#DFF0D8; 
                        padding:15px; 
                        margin-bottom:10px; 
                        border-radius:10px;
                        box-shadow: 2px 2px 5px gray;'>
                        <h3 style='margin:0; color:#3c763d;'>{i}: {title}</h3>
                        <p style='font-size:20px; margin:5px 0 0 0;'>{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            with tab3_col2:
                st.markdown(f"""
                    <div style='
                        background-color:#DFF0D8; 
                        padding:15px; 
                        margin-bottom:10px; 
                        border-radius:10px;
                        box-shadow: 2px 2px 5px gray;'>
                        <h3 style='margin:0; color:#3c763d;'>{i}: {title}</h3>
                        <p style='font-size:20px; margin:5px 0 0 0;'>{desc}</p>
                    </div>
                """, unsafe_allow_html=True)

    st.info(
        """**Important Note:** Our model only provides general recycling, composting, and trash recommendations based on common guidelines.
        Recycling rules vary by location, so check with local authorities for accuracy."""
    )

# About page | Provides more information about the app
with tab4:
    st.header("About", anchor=False)
    st.markdown("Smart waste disposal powered by AI.")

    with st.expander("**Why it is important**"):
        st.write("""
        In the United States alone, over **200 million tons of trash** were generated in 2018, and **146.1 million tons** of that ended up in landfills.
        
        Our mission is to **reduce this waste** by using AI to help people make smarter disposal decisions. With this app, users can experience how AI can be used for environmental good, while enjoying the fulfillment of reducing their environmental impact.
        
        #### 🌱 Impact
        
        - ♻️ **Reusing and recycling** reduces the need to extract raw natural resources like wood, water, and minerals.
        - ⚡ **Recycling saves energy** — for example, recycling just **10 plastic bottles** saves enough energy to power a laptop for **25 hours**.
        - 🗑️ **Recycling reduces landfill waste**, helping keep harmful materials out of our environment.
        
        > Even though it might feel small at first, every item recycled or reused is one less item that ends up in landfills.
        """)

        st.caption("📖 Source: ([EPA.gov](https://www.epa.gov/recycle/recycling-basics-and-benefits)).")

        st.page_link("https://www.epa.gov/recycle", label="Learn More >")

    with st.expander("**What it can Classify**"):
        st.markdown("""
        - Batteries and e-waste 
        - Food waste (Fruits, Vegetables, etc.)
        - Glass bottles and jars  
        - Brown cardboard and paper 
        - Clothing items
        - Lids, soda cans, aluminum cans, and containers
        - Plastic bottles, bags, and containers
        - Footwear
        - Masks, diapers, toothbrushes
        """)

    with st.expander("**What Makes Green Bin Different**"):
        st.write("""
        Most recycling apps rely on static databases. Green Bin uses real-time image classification
        and generative AI to give guidance on trash, compost, or recycling, all just from a photo!
        """)

    with st.expander("**Technology Behind the App**"):
        st.write("""
        - Feature extraction with **MobileNetV2** for image classification (**92%** accuracy)
        - **Gemini LLM** for context-aware recycling instructions
        - **Earth911 Search API** for drop-off locations based on zip code and item  
        - **Supabase** for the backend and data storage
        - Built with Python and Streamlit :streamlit: 
        """)

    with st.expander("**Data Source & License**"):
        st.write("- Contains information from [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification), which is made available here under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).")

    st.divider()

    st.markdown("**Contact Us**")
    st.markdown("Questions, feedback, or collaboration?")

    # Basic contact form using make.com
    def is_valid_email(email):
        # Basic regex pattern for email validation
        email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(email_pattern, email) is not None


    with st.form("contact_form"):
        name = st.text_input("First Name")
        email = st.text_input("Email Address")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Send")

    if submit_button:
        if not name:
            st.warning("Please provide your name.")
            st.stop()

        if not email:
            st.warning("Please provide your email address.")
            st.stop()

        if not is_valid_email(email):
            st.warning("Please provide a valid email address.")
            st.stop()

        if not message:
            st.warning("Please provide a message.")
            st.stop()

        data = {"email": email, "name": name, "message": message}
        response = requests.post(WEBHOOK_URL, json=data)

        if response.status_code == 200:
            st.success("Your message has been sent successfully! 🎉")
        else:
            st.error("There was an error sending your message.")
