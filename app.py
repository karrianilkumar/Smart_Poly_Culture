from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import difflib
import re

app = Flask(__name__)
CORS(app)

ROWS = 5  
COLS = 5  

# Define the plant matrix
plant_matrix = np.array([
    ["Be", "Ba", "Ca", "Bt", "To"],
    ["To", "Bt", "Ba", "Be", "Ca"],
    ["To", "Bt", "Be", "Ba", "Ca"],
    ["Bt", "Ba", "To", "Be", "Ca"],
    ["Ba", "Be", "Ca", "To", "Bt"]
])

# Matrix to track watering status (0 = OFF, 1 = DRIP, 2 = SHOWER)
water_status = np.zeros((ROWS, COLS), dtype=int)

# Plant Name Mapping
plant_map = {
    "basil": "Ba",
    "beans": "Be",
    "beetroot": "Bt",
    "carrot": "Ca",
    "tomato": "To",
    "all": "all"
}

# Plant synonyms for better recognition
PLANT_SYNONYMS = {
    "basil": "Ba", "herb": "Ba", "basils": "Ba",
    "beans": "Be", "bean": "Be", "green beans": "Be",
    "beetroot": "Bt", "beet": "Bt", "beets": "Bt",
    "carrot": "Ca", "carrots": "Ca",
    "tomato": "To", "tomatoes": "To", "tomoto": "To"
}

def classify_intent(text):
    """Simple rule-based intent classification"""
    text_lower = text.lower()

    # explicit phrase checks (higher priority)
    if re.search(r"\bturn\s+off\b", text_lower) or re.search(r"\bturn\s+of\b", text_lower):
        return "TurnOffWater"
    if re.search(r"\bturn\s+on\b", text_lower) or re.search(r"\bwater\b.*\bon\b", text_lower):
        return "TurnOnWater"

    # word-based keyword lists
    on_keywords = ['start', 'activate', 'open', 'begin', 'enable', 'on']
    off_keywords = ['stop', 'deactivate', 'close', 'end', 'disable', 'off']
    status_keywords = ['status', 'check', 'how are', 'what is', 'report']

    on_count = sum(1 for keyword in on_keywords if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower))
    off_count = sum(1 for keyword in off_keywords if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower))
    status_count = sum(1 for keyword in status_keywords if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower))

    if on_count > off_count and on_count > status_count:
        return "TurnOnWater"
    if off_count > on_count and off_count > status_count:
        return "TurnOffWater"
    if status_count > on_count and status_count > off_count:
        return "GetStatus"
    return "Unknown"


def parse_location(text):
    """Parse location like 'bed 2' or 'bed two' and return 0-based row index or None"""
    text_lower = text.lower()
    m = re.search(r"bed\s+(\d+)", text_lower)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < ROWS:
            return idx
    # support words one-two-three small set
    words_to_num = {'one':1,'two':2,'three':3,'four':4,'five':5}
    m2 = re.search(r"bed\s+(one|two|three|four|five)", text_lower)
    if m2:
        idx = words_to_num.get(m2.group(1), 1) - 1
        if 0 <= idx < ROWS:
            return idx
    return None


def parse_intensity(text):
    """Detect requested watering intensity: 0=OFF,1=DRIP,2=SHOWER"""
    text_lower = text.lower()
    shower_keywords = ['shower', 'heavy', 'high', 'fast', 'spray', 'bulk', 'lots']
    drip_keywords = ['drip', 'slow', 'light', 'gentle']

    if any(k in text_lower for k in shower_keywords):
        return 2
    if any(k in text_lower for k in drip_keywords):
        return 1
    # default when turning on is DRIP (1)
    return 1

def extract_plants(text):
    """Extract plant names from text using pattern matching"""
    text_lower = text.lower()
    # Tokenize input and look for exact or fuzzy matches
    tokens = re.findall(r"[a-zA-Z]+", text_lower)
    plants_found = []

    # Exact matches first
    for token in tokens:
        if token in PLANT_SYNONYMS:
            plant_code = PLANT_SYNONYMS[token]
            for original_name, code in plant_map.items():
                if code == plant_code and original_name.lower() != "all":
                    if original_name not in plants_found:
                        plants_found.append(original_name)
                    break

    # Fuzzy matching for misspellings (e.g., "tomoto")
    if not plants_found and tokens:
        synonyms_list = list(PLANT_SYNONYMS.keys())
        for token in tokens:
            close = difflib.get_close_matches(token, synonyms_list, n=1, cutoff=0.75)
            if close:
                plant_code = PLANT_SYNONYMS[close[0]]
                for original_name, code in plant_map.items():
                    if code == plant_code and original_name.lower() != "all":
                        if original_name not in plants_found:
                            plants_found.append(original_name)
                        break

    # Check for "all plants" keywords, but only use if no specific plant was found
    all_keywords = ['all', 'everything', 'every plant', 'whole garden', 'entire garden']
    if any(keyword in text_lower for keyword in all_keywords) and not plants_found:
        plants_found = ["all"]

    # If still no plants found, default to all (safe fallback)
    if not plants_found:
        plants_found = ["all"]

    return plants_found

def process_command(intent, plant_names, intensity=1, bed_index=None):
    """Process the command and update water status. intensity: 1=DRIP,2=SHOWER"""
    plant_codes = [plant_map.get(name, None) for name in plant_names]
    plant_codes = [code for code in plant_codes if code]

    if not plant_codes:
        return f"Sorry, I don't recognize those plants. Available plants: Basil, Beans, Beetroot, Carrot, Tomato."

    # pretty names for responses
    pretty = {k: k.capitalize() for k in plant_map.keys()}

    if intent == "TurnOnWater":
        if "all" in plant_codes:
            if bed_index is None:
                water_status.fill(intensity)
            else:
                water_status[bed_index, :] = intensity
            return f"Turning on water for all plants ({'SHOWER' if intensity==2 else 'DRIP'})."
        else:
            for plant_code in plant_codes:
                if bed_index is None:
                    water_status[plant_matrix == plant_code] = intensity
                else:
                    # restrict to the specified row
                    mask = (plant_matrix[bed_index, :] == plant_code)
                    water_status[bed_index, :][mask] = intensity
            plant_names_str = ', '.join([pretty.get(name.lower(), name) for name in plant_names if name != "all"])
            return f"Turning on water for {plant_names_str} ({'SHOWER' if intensity==2 else 'DRIP'})."

    elif intent == "TurnOffWater":
        if "all" in plant_codes:
            if bed_index is None:
                water_status.fill(0)
            else:
                water_status[bed_index, :] = 0
            return "Turning off water for all plants."
        else:
            for plant_code in plant_codes:
                if bed_index is None:
                    water_status[plant_matrix == plant_code] = 0
                else:
                    mask = (plant_matrix[bed_index, :] == plant_code)
                    water_status[bed_index, :][mask] = 0
            plant_names_str = ', '.join([pretty.get(name.lower(), name) for name in plant_names if name != "all"])
            return f"Turning off water for {plant_names_str}."

    elif intent == "GetStatus":
        # Return current watering status counts
        shower_count = np.sum(water_status == 2)
        drip_count = np.sum(water_status == 1)
        total_count = water_status.size
        return f"Current watering status: {drip_count} drip, {shower_count} shower out of {total_count} plants."

    else:
        return "Sorry, I didn't understand that command. Try 'turn on basil' or 'turn off all plants'."

# ===== ROUTES =====

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_status', methods=['GET'])
def get_status():
    """Returns the binary watering status matrix"""
    return jsonify(water_status.tolist())

@app.route('/ask', methods=['POST'])
def ask():
    """Chat endpoint"""
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        if not user_input:
            return jsonify({"reply": "Please send a message."})

        print(f"User message: {user_input}")

        # Use rule-based processing
        intent = classify_intent(user_input)
        plant_names = extract_plants(user_input)
        bed_index = parse_location(user_input)
        intensity = parse_intensity(user_input) if intent == 'TurnOnWater' else 0
        response_text = process_command(intent, plant_names, intensity, bed_index)

        return jsonify({"reply": response_text})

    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        return jsonify({"reply": "Sorry, I encountered an error."})

if __name__ == '__main__':
    print("Starting Smart Garden System...")
    print("Available endpoints:")
    print("  GET  /get_status    - Get current watering status")
    print("  POST /ask           - Chat interface")
    app.run(host='127.0.0.1', port=5000, debug=True)
