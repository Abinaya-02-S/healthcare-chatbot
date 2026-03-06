from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import csv
import re
import os
import json
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore")
import gspread
from oauth2client.service_account import ServiceAccountCredentials

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

SHEET_ID = "1VcCEX9Qfyj3krGh7JSiqM0lkHbE9sE-RiYufm7rF00s"

def save_to_sheet(name, age, gender, symptoms, disease, confidence):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_json = os.environ.get("GOOGLE_CREDENTIALS")
        if creds_json:
            creds_dict = json.loads(creds_json)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1
        sheet.append_row([name, age, gender, str(symptoms), disease, str(confidence)])
    except Exception as e:
        print(f"Google Sheets error: {e}")

# ------------------ Load Data ------------------
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# Fix disease name spaces
training['prognosis'] = training['prognosis'].str.strip()
testing['prognosis'] = testing['prognosis'].str.strip()

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

# Build disease symptom map
disease_symptom_map = {}
original_y = training['prognosis']
for disease in original_y.unique():
    rows = training[original_y == disease]
    sym_cols = rows.columns[:-1]
    freq = rows[sym_cols].mean()
    top_syms = freq[freq > 0.3].sort_values(ascending=False).index.tolist()
    disease_symptom_map[disease] = top_syms

# ------------------ Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def loadData():
    with open('MasterData/symptom_Description.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                description_list[row[0].strip()] = row[1]
    try:
        with open('MasterData/Symptom_severity.csv') as f:
            for row in csv.reader(f):
                try:
                    severityDictionary[row[0].strip()] = int(row[1])
                except:
                    pass
    except:
        with open('MasterData/symptom_severity.csv') as f:
            for row in csv.reader(f):
                try:
                    severityDictionary[row[0].strip()] = int(row[1])
                except:
                    pass
    with open('MasterData/symptom_precaution.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                precautionDictionary[row[0].strip()] = [row[1], row[2], row[3], row[4]]

loadData()

# ------------------ Symptom Extractor ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "fever": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
    "headache": "headache",
    "vomiting": "vomiting",
    "nausea": "nausea",
    "fatigue": "fatigue",
    "weakness": "fatigue",
    "itching": "itching",
    "rash": "skin_rash",
    "chest pain": "chest_pain",
    "back pain": "back_pain",
    "joint pain": "joint_pain",
    "dizziness": "dizziness",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence

def get_smart_followup(detected_symptoms, disease):
    disease_syms = disease_symptom_map.get(disease, [])
    followup = [s for s in disease_syms if s not in detected_symptoms]
    return followup[:6]

# ------------------ Doctor Dictionary ------------------
disease_doctor = {
    "Fungal infection": "Dermatologist",
    "Allergy": "General Physician",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Gastroenterologist",
    "Drug Reaction": "Dermatologist",
    "Peptic ulcer diseae": "Gastroenterologist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Gastroenterologist",
    "Malaria": "General Physician",
    "Chicken pox": "Dermatologist",
    "Dengue": "General Physician",
    "Typhoid": "General Physician",
    "hepatitis A": "Gastroenterologist",
    "Hepatitis B": "Gastroenterologist",
    "Hepatitis C": "Gastroenterologist",
    "Hepatitis D": "Gastroenterologist",
    "Hepatitis E": "Gastroenterologist",
    "Alcoholic hepatitis": "Gastroenterologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "General Physician",
    "Pneumonia": "Pulmonologist",
    "Dimorphic hemorrhoids(piles)": "Gastroenterologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthristis": "Orthopedist",
    "Arthritis": "Orthopedist",
    "(vertigo) Paroymsal  Positional Vertigo": "ENT Specialist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist",
    "AIDS": "General Physician"
}

doctor_contact = {
    "General Physician": "📞 +91 98765 43210",
    "Cardiologist": "📞 +91 91234 56789",
    "Dermatologist": "📞 +91 99887 76655",
    "Pulmonologist": "📞 +91 90909 80808",
    "Neurologist": "📞 +91 90000 11111",
    "Gastroenterologist": "📞 +91 95555 22222",
    "Endocrinologist": "📞 +91 93333 44444",
    "Orthopedist": "📞 +91 92222 55555",
    "Urologist": "📞 +91 91111 66666",
    "ENT Specialist": "📞 +91 94444 77777",
    "Vascular Surgeon": "📞 +91 96666 88888"
}

telemedicine_services = [
    "Government Telemedicine: eSanjeevani – https://esanjeevani.mohfw.gov.in",
    "Private Telemedicine: Apollo 24x7 / Practo",
    "Emergency Ambulance: 108"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_followup', methods=['POST'])
def get_followup():
    data = request.json
    symptoms_input = data.get('symptoms', '')
    detected = extract_symptoms(symptoms_input, cols)

    if not detected:
        return jsonify({"error": "No symptoms detected. Please describe your symptoms clearly. Example: I have fever and headache"})

    disease, confidence = predict_disease(detected)
    followup = get_smart_followup(detected, disease)

    return jsonify({
        "detected_symptoms": detected,
        "followup_symptoms": followup,
        "initial_disease": disease
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    name = data.get('name', 'User')
    age = data.get('age', '')
    gender = data.get('gender', '')
    symptom_list = data.get('symptom_list', [])

    if not symptom_list:
        return jsonify({"error": "No symptoms provided."})

    disease, confidence = predict_disease(symptom_list)
    doctor = disease_doctor.get(disease, "General Physician")
    doctor_phone = doctor_contact.get(doctor, "📞 +91 98765 43210")
    description = description_list.get(disease, 'No description available.')
    precautions = precautionDictionary.get(disease, [])

    save_to_sheet(name, age, gender, symptom_list, disease, confidence)

    return jsonify({
        "name": name,
        "disease": disease,
        "confidence": confidence,
        "doctor": doctor,
        "doctor_phone": doctor_phone,
        "description": description,
        "precautions": precautions,
        "telemedicine": telemedicine_services
    })

if __name__ == '__main__':
    app.run(debug=True)