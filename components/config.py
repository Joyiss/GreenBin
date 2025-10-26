import streamlit as st

gemini_api_key = st.secrets["GEMINI_API_KEY"]
earth911_api_key = st.secrets["EARTH911_API_KEY"]
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

news_api_key = st.secrets["NEWS_API_KEY"]

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

item_to_disposal_type = {
    "Battery": "Recycle",
    "Biological": "Compost",
    "Brown-glass": "Recycle",
    "Green-glass": "Recycle",
    "White-glass": "Recycle",
    "Cardboard": "Recycle",
    "Clothes": "Recycle",
    "Metal": "Recycle",
    "Paper": "Recycle",
    "Plastic": "Recycle",
    "Shoes": "Trash",
    "Trash": "Trash"
}

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
