"""
Crop Disease & Bio-Pesticide Assistant
=======================================
A Streamlit app that diagnoses plant disease from a leaf photo (optionally
supported by a short voice note describing symptoms), explains the disease
in plain language, and recommends a biological / organic remedy instead of
chemical pesticides.

Feature set:
- Multimodal diagnosis (image + optional voice note) via Gemini-Flash
- Weather-aware biocontrol timing (Open-Meteo, no API key needed)
- Low-cost DIY organic recipe for every diagnosis
- Economic impact estimate (yield loss % / treatment cost per acre)
- Dynamic acreage-based dosage calculator (kg of agent + litres of water)
- Multilingual output (7 languages) + localized advisory disclaimer
- WhatsApp sharing of the diagnosis summary
- Persistent "Scan History" tab (survives reloads via history.json)
- Multilingual follow-up chatbot
- Offline thermal-printer prescription export
- Community outbreak radar map (local CSV log + st.map)

Setup:
    pip install streamlit google-genai pillow audio-recorder-streamlit requests pandas

Run:
    streamlit run app1.py
"""

import streamlit as st
from PIL import Image
import io
import os
import re
import csv
import json
import base64
import random
import datetime
import textwrap
import urllib.parse

import requests
import pandas as pd

from google import genai
from google.genai import types
from audio_recorder_streamlit import audio_recorder


# =========================================================
# 1. CONFIGURATION
# =========================================================
# <-- PLACEHOLDER: replace with your key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
MODEL_NAME = "gemini-3.6-flash"

AUDIO_SAMPLE_RATE = 16000
AUDIO_BYTES_PER_SAMPLE = 2

DEFAULT_LOCATION_NAME = "Mumbai, India"
DEFAULT_LAT, DEFAULT_LON = 19.0760, 72.8777

OUTBREAK_LOG_PATH = "outbreak_log.csv"
OUTBREAK_LOG_COLUMNS = ["timestamp", "disease", "latitude", "longitude"]

HISTORY_PATH = "history.json"          # FEATURE 3: persistent scan history file

# --- FEATURE 1: dosage calculator fallback defaults (used only if the AI's
# numeric fields fail to parse) ---
DEFAULT_AGENT_KG_PER_ACRE = 1.5
DEFAULT_WATER_L_PER_ACRE = 175
BIGHA_TO_ACRE = 0.625   # approximate; varies by state -- shown as a note in the UI

# --- FEATURE 4: supported languages (label shown in UI -> name sent to the AI) ---
LANGUAGES = {
    "English": "English",
    "हिंदी (Hindi)": "Hindi",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "मराठी (Marathi)": "Marathi",
    "தமிழ் (Tamil)": "Tamil",
    "తెలుగు (Telugu)": "Telugu",
    "বাংলা (Bengali)": "Bengali",
}

# Fallback advisory text (English) -- used only if the AI response doesn't
# include a translated ADVISORY field for some reason.
FALLBACK_ADVISORY = (
    "⚠️ Advisory Notice & Caution: This diagnostic report is generated using "
    "AI-assisted biological analysis and is intended solely for informational "
    "and preliminary guidance. Biological effectiveness may vary with soil "
    "conditions, weather, and crop stages. Always handle biocontrol agents "
    "with appropriate safety gear, follow package instructions, and consult "
    "your local agricultural extension officer (Krishi Vigyan Kendra / Ag "
    "specialist) before large-scale application."
)


# =========================================================
# 2. LOCAL BIO-PESTICIDE DATABASE (grounding reference)
# =========================================================
BIO_PESTICIDE_DB = {
    "Tomato Late Blight": {
        "agent": "Bacillus subtilis (strain QST 713)",
        "mechanism": (
            "Colonizes the leaf surface and root zone ahead of the pathogen, "
            "competing for space and nutrients, and secretes lipopeptides "
            "that disrupt the cell membrane of Phytophthora infestans spores."
        ),
        "application": (
            "Mix 5 g/L of Bacillus subtilis wettable powder in water. "
            "Spray every 7-10 days, starting at first sign of disease."
        ),
    },
    "Tomato Early Blight": {
        "agent": "Trichoderma harzianum",
        "mechanism": (
            "Acts through mycoparasitism -- directly attacks and digests "
            "the hyphae of Alternaria solani, and induces systemic "
            "resistance in the host plant."
        ),
        "application": (
            "Soil drench/spray at 4 g/L at transplanting, then spray on "
            "leaves every 10-14 days."
        ),
    },
    "Rice Blast": {
        "agent": "Pseudomonas fluorescens",
        "mechanism": (
            "Produces siderophores that starve the pathogen of iron and "
            "triggers induced systemic resistance in rice plants."
        ),
        "application": (
            "Seed treatment: 10 g/kg of seed before sowing. Foliar spray: "
            "5 g/L at tillering and panicle initiation."
        ),
    },
    "Wheat Powdery Mildew": {
        "agent": "Ampelomyces quisqualis",
        "mechanism": (
            "A mycoparasite that infects and destroys the fungal structures "
            "of powdery mildew fungi, collapsing them from within."
        ),
        "application": "Spray at 2-3 mL/L at first sign of white powdery patches.",
    },
}


# =========================================================
# 3. WEATHER (Open-Meteo, no API key required)
# =========================================================
def geocode_location(place_name: str):
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place_name, "count": 1, "language": "en"},
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        top = results[0]
        display_name = f"{top.get('name')}, {top.get('country', '')}".strip(
            ", ")
        return top["latitude"], top["longitude"], display_name
    except Exception:
        return None


def fetch_weather(lat: float, lon: float):
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m",
                "timezone": "auto",
            },
            timeout=8,
        )
        resp.raise_for_status()
        current = resp.json().get("current", {})
        if "temperature_2m" not in current:
            return None
        return {
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
        }
    except Exception:
        return None


# =========================================================
# 4. OUTBREAK RADAR — LOCAL CSV LOGGING
# =========================================================
def log_outbreak(disease: str, lat: float, lon: float):
    jitter_lat = lat + random.uniform(-0.05, 0.05)
    jitter_lon = lon + random.uniform(-0.05, 0.05)
    file_exists = os.path.exists(OUTBREAK_LOG_PATH)
    with open(OUTBREAK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(OUTBREAK_LOG_COLUMNS)
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            disease, jitter_lat, jitter_lon,
        ])


def load_outbreak_log() -> pd.DataFrame:
    if not os.path.exists(OUTBREAK_LOG_PATH):
        return pd.DataFrame(columns=OUTBREAK_LOG_COLUMNS)
    try:
        df = pd.read_csv(OUTBREAK_LOG_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=OUTBREAK_LOG_COLUMNS)


# =========================================================
# 5. FEATURE 3: PERSISTENT SCAN HISTORY (history.json)
# =========================================================
def load_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history_list: list):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # best-effort; history still lives in session_state for this run


def image_bytes_to_thumbnail_b64(image_bytes: bytes, size=(160, 160)) -> str:
    """Shrinks the image and returns a base64 JPEG string, small enough to
    store cheaply inside history.json alongside the diagnosis text."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def add_history_entry(diagnosis: dict, image_bytes: bytes, farm_acres: float, language_label: str):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "thumbnail_b64": image_bytes_to_thumbnail_b64(image_bytes),
        "disease": diagnosis.get("DISEASE", "N/A"),
        "severity": diagnosis.get("SEVERITY", "N/A"),
        "language": language_label,
        "farm_acres": farm_acres,
        # full structured dict -> lets "View Details" reload instantly, no API call
        "diagnosis": diagnosis,
    }
    st.session_state.history_list.insert(0, entry)   # newest first
    save_history(st.session_state.history_list)


# =========================================================
# 6. DIAGNOSIS SYSTEM PROMPT (biological focus, multilingual, dosage math)
# =========================================================
DIAGNOSTICIAN_SYSTEM_PROMPT_TEMPLATE = """
You are Dr. Kisan-AI, an agricultural biotechnologist specializing in
biological crop protection for Indian smallholder farms. Respond ENTIRELY
in {language_name}, using its native script. Keep scientific species names
in English/Latin (e.g. "Trichoderma harzianum") even when the surrounding
sentence is in {language_name}.

You are helping diagnose plant disease from a photo (and, if provided, a
farmer's voice description of symptoms, and current local weather).

Your task:
1. Identify the crop and the most likely disease.
2. Rate CONFIDENCE (High/Medium/Low) in this diagnosis, and separately rate
   SEVERITY (Low/Medium/High/Critical) -- how advanced the visible infection
   looks.
3. Briefly explain what the disease IS: cause, symptoms, how it spreads.
4. Recommend a BIOLOGICAL / organic control ONLY -- never a synthetic
   chemical pesticide or fungicide. Name a specific biocontrol species and
   strain where possible.
5. Give a DIY_RECIPE: a traditional homemade preparation using common local
   ingredients (buttermilk/chaas, neem, cow urine, garlic-chili,
   panchagavya, etc.), with rough proportions and method.
6. Give the STANDARD per-acre dosage rate for the recommended agent as a
   single representative NUMBER (not a range): AGENT_DOSE_KG_PER_ACRE (kg of
   formulation per acre) and WATER_L_PER_ACRE (litres of water to dilute in,
   per acre). Also state APPLICATION_TECHNIQUE (foliar spray / soil
   drenching / seed treatment).
7. If current weather data is provided, evaluate whether it is suitable for
   applying LIVE biological agents right now, in WEATHER_ADVICE.
8. Estimate YIELD_LOSS_PCT if untreated (a range) and TREATMENT_COST_PER_ACRE
   in Indian Rupees (a range).
9. ADVISORY: translate the following caution notice faithfully into
   {language_name}, keeping its FULL meaning -- do not shorten or summarize it:
   "Advisory Notice and Caution: This diagnostic report is generated using
   AI-assisted biological analysis and is intended solely for informational
   and preliminary guidance. Biological effectiveness may vary with soil
   conditions, weather, and crop stages. Always handle biocontrol agents
   with appropriate safety gear, follow package instructions, and consult
   your local agricultural extension officer (Krishi Vigyan Kendra / Ag
   specialist) before large-scale application."

Respond in this EXACT structure (plain text, no markdown symbols). Keep the
field labels below in English exactly as written, but write the VALUES in
{language_name}:

DISEASE: <Crop> <Disease name>
CONFIDENCE: <High/Medium/Low>
SEVERITY: <Low/Medium/High/Critical>
ABOUT: <2-3 sentences>
BIO_AGENT: <specific organism/strain or organic remedy>
MECHANISM: <1-2 sentences>
DOSAGE: <concrete application instructions in words>
DIY_RECIPE: <homemade preparation, ingredients + method>
AGENT_DOSE_KG_PER_ACRE: <number only, e.g. 1.5>
WATER_L_PER_ACRE: <number only, e.g. 175>
APPLICATION_TECHNIQUE: <foliar spray / soil drenching / seed treatment>
WEATHER_ADVICE: <advice given the weather provided, or note if none was given>
YIELD_LOSS_PCT: <range, e.g. "20-30%">
TREATMENT_COST_PER_ACRE: <range in Rs, e.g. "Rs 400-700/acre">
ADVISORY: <the translated caution notice, in full>
NOTES: <extra field advice>
"""

DIAGNOSIS_FIELDS = [
    "DISEASE", "CONFIDENCE", "SEVERITY", "ABOUT", "BIO_AGENT", "MECHANISM", "DOSAGE",
    "DIY_RECIPE", "AGENT_DOSE_KG_PER_ACRE", "WATER_L_PER_ACRE", "APPLICATION_TECHNIQUE",
    "WEATHER_ADVICE", "YIELD_LOSS_PCT", "TREATMENT_COST_PER_ACRE", "ADVISORY", "NOTES",
]


def parse_structured_diagnosis(raw_text: str) -> dict:
    result = {f: "" for f in DIAGNOSIS_FIELDS}
    current_field = None
    for line in raw_text.splitlines():
        stripped = line.strip()
        matched = False
        for f in DIAGNOSIS_FIELDS:
            if stripped.upper().startswith(f + ":"):
                current_field = f
                result[f] = stripped[len(f) + 1:].strip()
                matched = True
                break
        if not matched and current_field:
            result[current_field] += " " + stripped
    return result


def extract_first_number(text: str, default: float) -> float:
    """Pulls the first numeric value out of a string like '1.5 kg' -> 1.5.
    Falls back to `default` if nothing numeric is found."""
    if not text:
        return default
    match = re.search(r"\d+\.?\d*", text)
    return float(match.group()) if match else default


# =========================================================
# 7. GEMINI CALLS
# =========================================================
def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def identify_disease(image_bytes: bytes, api_key: str, language_name: str,
                     audio_bytes: bytes | None = None,
                     weather: dict | None = None) -> dict:
    client = get_client(api_key)

    parts = [types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]

    if audio_bytes:
        parts.append(types.Part.from_bytes(
            data=audio_bytes, mime_type="audio/wav"))
        parts.append(
            "The audio clip above is the farmer describing the crop's "
            "symptoms in their own words/language. Use it together with "
            "the image to refine your diagnosis."
        )

    if weather:
        parts.append(
            f"Current weather at the farm location: "
            f"{weather['temperature_c']} degrees C, "
            f"{weather['humidity_pct']}% relative humidity. "
            f"Use this for your WEATHER_ADVICE field."
        )

    prompt = DIAGNOSTICIAN_SYSTEM_PROMPT_TEMPLATE.format(
        language_name=language_name)
    parts.append(prompt)

    response = client.models.generate_content(model=MODEL_NAME, contents=parts)

    raw_text = response.text.strip()
    parsed = parse_structured_diagnosis(raw_text)
    parsed["_raw"] = raw_text
    if not parsed.get("ADVISORY"):
        parsed["ADVISORY"] = FALLBACK_ADVISORY
    return parsed


def match_disease_to_db(disease_name: str) -> dict | None:
    for key in BIO_PESTICIDE_DB:
        if key.lower() == disease_name.lower():
            return {"matched_name": key, **BIO_PESTICIDE_DB[key]}
    for key in BIO_PESTICIDE_DB:
        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
            return {"matched_name": key, **BIO_PESTICIDE_DB[key]}
    return None


CHATBOT_SYSTEM_PROMPT = """
You are Dr. Kisan-AI, the same agricultural biotechnologist who produced
the diagnosis below. You are now in a follow-up conversation with the
farmer or a village coordinator about this SAME plant image and diagnosis.

Diagnosis context:
{diagnosis_context}

Rules:
- Detect the language of the user's latest message (Hindi, Marathi, Tamil,
  Telugu, Bengali, Kannada, Punjabi, romanized Hindi/"Hinglish", English,
  or any other Indian regional language) and reply in that SAME language.
- Keep answers practical and short (2-5 sentences), focused on
  biological/organic remedies -- never recommend chemical pesticides.
- You may refer back to the uploaded image if the question is visual.
"""


def chat_with_agronomist(image_bytes: bytes, diagnosis_context: str,
                         chat_history: list, api_key: str) -> str:
    client = get_client(api_key)
    system_prompt = CHATBOT_SYSTEM_PROMPT.format(
        diagnosis_context=diagnosis_context)

    contents = [types.Part.from_bytes(
        data=image_bytes, mime_type="image/jpeg")]
    for msg in chat_history:
        role_label = "Farmer" if msg["role"] == "user" else "Dr. Kisan-AI"
        contents.append(f"{role_label}: {msg['content']}")
    contents.append(system_prompt)

    response = client.models.generate_content(
        model=MODEL_NAME, contents=contents)
    return response.text.strip()


# =========================================================
# 8. THERMAL RECEIPT FORMATTER
# =========================================================
def build_thermal_receipt(diagnosis: dict, printer_width: int = 32,
                          total_agent_kg: float | None = None,
                          total_water_l: float | None = None,
                          farm_acres: float | None = None) -> str:
    line = "-" * printer_width

    def wrap(label: str, value: str) -> str:
        wrapped = textwrap.wrap(value, width=printer_width) or [""]
        out = [f"{label}:"] if label else []
        out.extend(wrapped)
        return "\n".join(out)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    receipt_lines = [
        "BIO-PRESCRIPTION".center(printer_width),
        line,
        f"Date: {timestamp}".ljust(printer_width),
        line,
        wrap("DIAGNOSIS", diagnosis.get("DISEASE", "N/A")),
        f"Confidence: {diagnosis.get('CONFIDENCE', 'N/A')}  Severity: {diagnosis.get('SEVERITY', 'N/A')}",
        line,
        wrap("BIO-AGENT", diagnosis.get("BIO_AGENT", "N/A")),
        "",
        wrap("HOW IT WORKS", diagnosis.get("MECHANISM", "N/A")),
        line,
    ]

    # --- FEATURE 1: calculated dosage totals, if provided ---
    if total_agent_kg is not None and total_water_l is not None and farm_acres is not None:
        receipt_lines += [
            wrap("DOSAGE CALCULATOR", f"For {farm_acres:g} acre(s):"),
            f"Bio-agent: {total_agent_kg:.2f} kg",
            f"Water: {total_water_l:.0f} L",
            f"Method: {diagnosis.get('APPLICATION_TECHNIQUE', 'N/A')}",
            line,
        ]

    receipt_lines += [
        wrap("DIY RECIPE", diagnosis.get("DIY_RECIPE", "N/A")),
        line,
        wrap("WEATHER ADVICE", diagnosis.get("WEATHER_ADVICE", "N/A")),
        line,
        f"Est. yield loss if untreated: {diagnosis.get('YIELD_LOSS_PCT', 'N/A')}",
        f"Treatment cost: {diagnosis.get('TREATMENT_COST_PER_ACRE', 'N/A')}",
        line,
        wrap("NOTES", diagnosis.get("NOTES", "-")),
        line,
        wrap("", diagnosis.get("ADVISORY", FALLBACK_ADVISORY)),
        line,
    ]
    return "\n".join(receipt_lines)


# =========================================================
# 9. SHARED RENDER HELPER — used by BOTH the live Diagnosis tab
#    and the "View Details" section in Scan History (FEATURE 3),
#    so history entries display instantly with no extra API call.
# =========================================================
def render_diagnosis_card(d: dict, key_prefix: str, default_farm_acres: float = 1.0):
    st.subheader(d.get("DISEASE", "N/A"))
    confidence = d.get("CONFIDENCE", "N/A")
    severity = d.get("SEVERITY", "N/A")
    conf_badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
    sev_badge = {"Low": "🟢", "Medium": "🟡",
                 "High": "🟠", "Critical": "🔴"}.get(severity, "⚪")
    st.caption(
        f"{conf_badge} Confidence: {confidence}   ·   {sev_badge} Severity: {severity}")

    st.markdown("**About this disease**")
    st.write(d.get("ABOUT", "N/A"))

    m1, m2 = st.columns(2)
    m1.metric("📉 Est. yield loss if untreated", d.get("YIELD_LOSS_PCT", "N/A"))
    m2.metric("💰 Bio-treatment cost / acre",
              d.get("TREATMENT_COST_PER_ACRE", "N/A"))

    weather_advice = d.get("WEATHER_ADVICE", "")
    if weather_advice:
        st.markdown("**🌦️ Weather-aware application advice**")
        st.info(weather_advice)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🦠 Biological control agent**")
        st.success(d.get("BIO_AGENT", "N/A"))
        st.markdown("**⚙️ How it works**")
        st.write(d.get("MECHANISM", "N/A"))
    with col2:
        st.markdown("**📋 Dosage & application**")
        st.write(d.get("DOSAGE", "N/A"))
        st.markdown("**📝 Extra notes**")
        st.write(d.get("NOTES", "-"))

    with st.expander("🧑‍🌾 Low-cost DIY organic recipe", expanded=True):
        st.write(d.get("DIY_RECIPE", "N/A"))

    # ===================================================
    # FEATURE 1: Dynamic Acreage Dosage Calculator
    # ===================================================
    st.markdown("### 🧮 Dosage Calculator")
    calc_col1, calc_col2 = st.columns([2, 1])
    with calc_col1:
        farm_size = st.number_input(
            "Enter Farm Size", min_value=0.1, value=default_farm_acres, step=0.1,
            key=f"{key_prefix}_farm_size",
        )
    with calc_col2:
        unit = st.selectbox(
            "Unit", ["Acres", "Bigha (approx.)"], key=f"{key_prefix}_unit"
        )
    farm_acres = farm_size if unit == "Acres" else farm_size * BIGHA_TO_ACRE
    if unit != "Acres":
        st.caption(
            f"≈ {farm_acres:.2f} acres (1 Bigha ≈ {BIGHA_TO_ACRE} acre — varies by state)")

    agent_rate = extract_first_number(
        d.get("AGENT_DOSE_KG_PER_ACRE", ""), DEFAULT_AGENT_KG_PER_ACRE)
    water_rate = extract_first_number(
        d.get("WATER_L_PER_ACRE", ""), DEFAULT_WATER_L_PER_ACRE)
    total_agent_kg = agent_rate * farm_acres
    total_water_l = water_rate * farm_acres

    with st.container(border=True):
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("🧪 Bio-Agent Required", f"{total_agent_kg:.2f} kg")
        cc2.metric("💧 Water Required", f"{total_water_l:.0f} L")
        cc3.metric("🚿 Technique", d.get("APPLICATION_TECHNIQUE", "N/A"))

    # ===================================================
    # FEATURE 4: WhatsApp sharing
    # ===================================================
    share_text = (
        f"🌱 Crop Diagnosis Report\n"
        f"Disease: {d.get('DISEASE', 'N/A')}\n"
        f"Bio-Agent: {d.get('BIO_AGENT', 'N/A')}\n"
        f"Dosage: {total_agent_kg:.2f} kg in {total_water_l:.0f} L water "
        f"for {farm_acres:.2f} acre(s) ({d.get('APPLICATION_TECHNIQUE', 'N/A')})\n"
        f"Est. cost: {d.get('TREATMENT_COST_PER_ACRE', 'N/A')}\n\n"
        f"{d.get('ADVISORY', FALLBACK_ADVISORY)}"
    )
    whatsapp_url = "https://wa.me/?text=" + urllib.parse.quote(share_text)
    st.link_button("📤 Share Summary via WhatsApp",
                   whatsapp_url, use_container_width=True)

    if d.get("_db_reference"):
        with st.expander("📚 Reference database entry"):
            ref = d["_db_reference"]
            st.write(f"**Matched entry:** {ref['matched_name']}")
            st.write(f"**Agent:** {ref['agent']}")
            st.write(f"**Mechanism:** {ref['mechanism']}")
            st.write(f"**Application:** {ref['application']}")

    # ===================================================
    # FEATURE 2: Advisory / caution banner
    # ===================================================
    st.warning(d.get("ADVISORY", FALLBACK_ADVISORY))

    return farm_acres, total_agent_kg, total_water_l


# =========================================================
# 10. PAGE CONFIG + STYLING
# =========================================================
st.set_page_config(
    page_title="Crop Disease & Bio-Pesticide Assistant",
    page_icon="🌱",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stChatMessage"] { border-radius: 12px; }
    .stButton>button, .stLinkButton>a { border-radius: 8px; font-weight: 600; }
    .stDownloadButton>button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 11. FEATURE 4: SIDEBAR — LANGUAGE SELECTOR
# =========================================================
st.sidebar.header("⚙️ Settings")
language_label = st.sidebar.selectbox(
    "🌐 Output language", list(LANGUAGES.keys()))
language_name = LANGUAGES[language_label]
st.sidebar.caption(
    "Diagnosis, DIY recipe, dosage notes and the caution notice will all be returned in this language.")

st.title("🌱 Crop Disease & Bio-Pesticide Assistant")
st.caption(
    "Upload a leaf photo to get an instant biological diagnosis and treatment plan.")

# --- session_state initialisation --------------------------------------
defaults = {
    "diagnosis": None,
    "image_bytes": None,
    "chat_history": [],
    "recorder_key": 0,
    "location_name": DEFAULT_LOCATION_NAME,
    "location_coords": (DEFAULT_LAT, DEFAULT_LON),
    "weather": None,
    "history_list": load_history(),   # FEATURE 3: loaded once from history.json
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 12. INPUT SECTION
# =========================================================
col_img, col_audio = st.columns([1.3, 1])

with col_img:
    st.markdown("#### 📷 Leaf photo")
    uploaded_file = st.file_uploader(
        "Upload a clear photo of the affected leaf",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

with col_audio:
    st.markdown("#### 🎙️ Describe symptoms (optional)")
    st.caption("Any language. Click once to start, click again to stop.")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e63946",
        neutral_color="#2a9d8f",
        icon_size="3x",
        key=f"recorder_{st.session_state.recorder_key}",
    )
    if audio_bytes:
        est_seconds = len(audio_bytes) / \
            (AUDIO_SAMPLE_RATE * AUDIO_BYTES_PER_SAMPLE)
        if est_seconds < 0.6:
            st.warning(
                f"Only ~{est_seconds:.1f}s captured — click the mic again and stop it properly.")
        else:
            st.success(f"🎧 Recorded ~{est_seconds:.1f} seconds")
        st.audio(audio_bytes, format="audio/wav")
        if st.button("🗑️ Clear recording", use_container_width=True):
            st.session_state.recorder_key += 1
            st.rerun()

st.markdown("#### 📍 Farm location (for weather-aware advice)")
loc_col1, loc_col2 = st.columns([3, 1])
with loc_col1:
    location_input = st.text_input(
        "Location", value=st.session_state.location_name, label_visibility="collapsed"
    )
with loc_col2:
    fetch_weather_clicked = st.button(
        "Update weather", use_container_width=True)

if fetch_weather_clicked:
    with st.spinner("Looking up location and weather..."):
        geo = geocode_location(location_input)
        if geo is None:
            st.warning(
                "Couldn't resolve that location — using the previous one.")
        else:
            lat, lon, resolved_name = geo
            st.session_state.location_name = resolved_name
            st.session_state.location_coords = (lat, lon)
            st.session_state.weather = fetch_weather(lat, lon)
            if st.session_state.weather is None:
                st.warning(
                    "Location found, but weather lookup failed — try again shortly.")

if st.session_state.weather:
    w = st.session_state.weather
    wcol1, wcol2 = st.columns(2)
    wcol1.metric("🌡️ Temperature", f"{w['temperature_c']} °C")
    wcol2.metric("💧 Humidity", f"{w['humidity_pct']}%")
    st.caption(f"Weather for: {st.session_state.location_name}")

st.divider()

diagnose_disabled = uploaded_file is None
if st.button("🔍 Diagnose Disease", type="primary", disabled=diagnose_disabled, use_container_width=True):
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        st.error(
            "⚠️ Add your Gemini API key to `GEMINI_API_KEY` at the top of the script.")
    else:
        with st.spinner("Analyzing image, audio, and weather..."):
            try:
                img_byte_arr = io.BytesIO()
                image.convert("RGB").save(img_byte_arr, format="JPEG")
                img_bytes = img_byte_arr.getvalue()

                if st.session_state.weather is None:
                    geo = geocode_location(st.session_state.location_name)
                    if geo:
                        lat, lon, resolved_name = geo
                        st.session_state.location_name = resolved_name
                        st.session_state.location_coords = (lat, lon)
                        st.session_state.weather = fetch_weather(lat, lon)

                diagnosis = identify_disease(
                    img_bytes, GEMINI_API_KEY, language_name,
                    audio_bytes=audio_bytes,
                    weather=st.session_state.weather,
                )

                db_match = match_disease_to_db(diagnosis.get("DISEASE", ""))
                if db_match:
                    diagnosis["_db_reference"] = db_match

                st.session_state.diagnosis = diagnosis
                st.session_state.image_bytes = img_bytes
                st.session_state.chat_history = []

                lat, lon = st.session_state.location_coords
                log_outbreak(diagnosis.get("DISEASE", "Unknown"), lat, lon)

                # --- FEATURE 3: save to persistent scan history ---
                add_history_entry(diagnosis, img_bytes,
                                  default_farm_acres := 1.0, language_label)

            except Exception as e:
                st.error(f"Something went wrong calling the Gemini API: {e}")

if uploaded_file is None:
    st.info("👆 Upload a photo to enable diagnosis.")


# =========================================================
# 13. RESULTS — TABBED LAYOUT
# =========================================================
if st.session_state.diagnosis:
    d = st.session_state.diagnosis
    st.divider()

    tab_diagnosis, tab_chat, tab_receipt, tab_radar, tab_history = st.tabs(
        ["🧪 Diagnosis", "💬 Ask a question", "🖨️ Prescription",
            "🗺️ Outbreak Radar", "📜 Scan History"]
    )

    # --- Diagnosis tab (uses the shared render helper) ---------------------
    with tab_diagnosis:
        farm_acres, total_agent_kg, total_water_l = render_diagnosis_card(
            d, key_prefix="live")

    # --- Chat tab -----------------------------------------------------------
    with tab_chat:
        st.caption("Type in any language — the reply will match it.")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input(
            "e.g. Is this bio-pesticide safe for bees?")
        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = chat_with_agronomist(
                            st.session_state.image_bytes,
                            d.get("_raw", ""),
                            st.session_state.chat_history,
                            GEMINI_API_KEY,
                        )
                        st.markdown(reply)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Chat error: {e}")

    # --- Prescription tab -----------------------------------------------------
    with tab_receipt:
        printer_choice = st.radio("Printer width", options=[
                                  "58mm", "80mm"], horizontal=True)
        width = 32 if printer_choice == "58mm" else 48
        receipt_text = build_thermal_receipt(
            d, printer_width=width,
            total_agent_kg=total_agent_kg, total_water_l=total_water_l, farm_acres=farm_acres,
        )
        st.code(receipt_text, language=None)
        st.download_button(
            "⬇️ Download prescription.txt",
            data=receipt_text,
            file_name="bio_prescription.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.caption(
            "Send this file to any Bluetooth thermal-printer app on Android (e.g. RawBT).")
        st.warning(d.get("ADVISORY", FALLBACK_ADVISORY))

    # --- Outbreak Radar tab -------------------------------------------
    with tab_radar:
        st.caption(
            "Every diagnosis run on this device is logged with an "
            "approximate location, building a simple community early-warning "
            "map of disease hotspots."
        )
        log_df = load_outbreak_log()
        if log_df.empty:
            st.info("No diagnoses logged yet. Run a diagnosis to start the radar.")
        else:
            map_df = log_df.rename(
                columns={"latitude": "lat", "longitude": "lon"})
            st.map(map_df[["lat", "lon"]], size=20)

            st.markdown("**Recent diagnoses**")
            recent = log_df.sort_values("timestamp", ascending=False).head(20)
            st.dataframe(recent[["timestamp", "disease"]],
                         use_container_width=True, hide_index=True)

            st.markdown("**Most common diseases logged**")
            counts = log_df["disease"].value_counts().reset_index()
            counts.columns = ["Disease", "Reports"]
            st.dataframe(counts, use_container_width=True, hide_index=True)

    # ===================================================
    # FEATURE 3: Scan History tab
    # ===================================================
    with tab_history:
        st.caption("Past diagnoses on this device, saved to `history.json`. 'View Details' reloads the full report instantly — no API call needed.")

        if not st.session_state.history_list:
            st.info("No scans yet. Diagnose a leaf to start building history.")
        else:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history_list = []
                save_history([])
                st.rerun()

            for i, entry in enumerate(st.session_state.history_list):
                sev = entry.get("severity", "N/A")
                sev_badge = {"Low": "🟢", "Medium": "🟡",
                             "High": "🟠", "Critical": "🔴"}.get(sev, "⚪")
                header = f"{sev_badge} {entry.get('disease', 'N/A')} — {entry.get('timestamp', '')}"
                with st.expander(header):
                    thumb_col, info_col = st.columns([1, 3])
                    with thumb_col:
                        try:
                            thumb_bytes = base64.b64decode(
                                entry["thumbnail_b64"])
                            st.image(thumb_bytes, use_container_width=True)
                        except Exception:
                            st.caption("(image unavailable)")
                    with info_col:
                        st.write(f"**Severity:** {sev}")
                        st.write(
                            f"**Language:** {entry.get('language', 'N/A')}")
                        st.write(
                            f"**Farm size on record:** {entry.get('farm_acres', 'N/A')} acre(s)")

                    view_key = f"view_{i}"
                    if st.button("🔍 View Details", key=view_key):
                        st.session_state[f"expanded_{i}"] = True

                    if st.session_state.get(f"expanded_{i}"):
                        st.divider()
                        render_diagnosis_card(
                            entry["diagnosis"],
                            key_prefix=f"hist_{i}",
                            default_farm_acres=entry.get("farm_acres", 1.0),
                        )
