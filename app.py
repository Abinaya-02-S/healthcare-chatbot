from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import csv
import re
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# ------------------ Load Data ------------------
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

# ------------------ Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def loadData():
    with open('MasterData/symptom_Description.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                description_list[row[0]] = row[1]
    try:
        with open('MasterData/Symptom_severity.csv') as f:
            for row in csv.reader(f):
                try:
                    severityDictionary[row[0]] = int(row[1])
                except:
                    pass
    except:
        with open('MasterData/symptom_severity.csv') as f:
            for row in csv.reader(f):
                try:
                    severityDictionary[row[0]] = int(row[1])
                except:
                    pass
    with open('MasterData/symptom_precaution.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

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

# ------------------ Doctor Dictionary ------------------
disease_doctor = {
    "Fungal infection": "Dermatologist",
    "Allergy": "General Physician",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Gastroenterologist",
    "Drug Reaction": "Dermatologist",
    "Peptic ulcer disease": "Gastroenterologist",
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
    "Dimorphic hemmorhoids(piles)": "Gastroenterologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthritis": "Orthopedist",
    "Arthritis": "Orthopedist",
    "(Vertigo) Paroxysmal Positional Vertigo": "ENT Specialist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist"
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

# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_followup', methods=['POST'])
def get_followup():
    data = request.json
    symptoms_input = data.get('symptoms', '')
    detected = extract_symptoms(symptoms_input, cols)

    if not detected:
        return jsonify({"error": "No symptoms detected. Please describe more clearly."})

    disease, _ = predict_disease(detected)
    
    disease_rows = training[training['prognosis'] == disease]
    if not disease_rows.empty:
        disease_syms = list(disease_rows.iloc[0][:-1].index[
            disease_rows.iloc[0][:-1] == 1
        ])
        followup = [s for s in disease_syms if s not in detected][:6]
    else:
        followup = []

    return jsonify({
        "detected_symptoms": detected,
        "followup_symptoms": followup
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

    with open("patient_logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, age, gender, symptom_list, disease, confidence])

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