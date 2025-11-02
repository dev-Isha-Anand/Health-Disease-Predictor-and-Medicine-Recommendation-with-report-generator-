from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import re
import difflib
from pymongo import MongoClient
import math

app = Flask(__name__)

# ✅ MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['health_app']
user_collection = db['user_data']

# ✅ Load datasets and model
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/disease_specific_yoga_workout_all_42.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/workout_df.csv")

svc = pickle.load(open('svc.pkl', 'rb'))

# ✅ Dictionaries
symptoms_dict = { ... }   # keep your existing dictionary
diseases_list = { ... }   # keep your existing disease mapping
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
_symptom_keys = list(symptoms_dict.keys())



# ✅ Helper functions (same as yours)
def normalize_symptom(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r'[\s\-]+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s)
    return s

def match_symptom_key(user_symptom: str, cutoff=0.7):
    if not user_symptom:
        return None
    norm = normalize_symptom(user_symptom)
    if norm in symptoms_dict:
        return norm
    manual_map = {'skinrash': 'skin_rash', 'spottingurination': 'spotting_ urination', 'coldhandsandfeets': 'cold_hands_and_feets', 'yellowishskin': 'yellowish_skin', 'highfever': 'high_fever', 'mildfever': 'mild_fever', 'stomachpain': 'stomach_pain'}
    if norm in manual_map:
        return manual_map[norm]
    keys_for_match = _symptom_keys + [k.replace('_', '') for k in _symptom_keys]
    matches = difflib.get_close_matches(norm, keys_for_match, n=1, cutoff=cutoff)
    if matches:
        candidate = matches[0]
        if candidate in symptoms_dict:
            return candidate
        else:
            for k in _symptom_keys:
                if k.replace('_', '') == candidate:
                    return k
    return None

def helper(dis):
    disease_col = 'Disease' if 'Disease' in description.columns else 'disease'
    desc = description[description[disease_col] == dis]['Description']
    desc = " ".join([w for w in desc]) if not desc.empty else "No description available."
    pre = precautions[precautions[disease_col] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre_list = [col for col in pre.values.flatten() if pd.notna(col)]
    med = medications[medications[disease_col] == dis]['Medication']
    med_list = [m for m in med.values if pd.notna(m)]
    die = diets[diets[disease_col] == dis]['workout']
    die_list = [d for d in die.values if pd.notna(d)]
    if 'disease' in workout.columns:
        wrkout = workout[workout['disease'] == dis]['workout']
    else:
        wrkout = workout[workout[disease_col] == dis]['workout'] if disease_col in workout.columns else []
    wrkout_list = [w for w in wrkout.values if pd.notna(w)]
    return desc, pre_list, med_list, die_list, wrkout_list

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    recognized = []
    for raw in patient_symptoms:
        matched_key = match_symptom_key(raw)
        if matched_key:
            recognized.append(matched_key)
            input_vector[symptoms_dict[matched_key]] = 1
    if not recognized:
        raise ValueError("No recognizable symptoms provided.")
    input_df = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    pred_index = svc.predict(input_df)[0]
    return diseases_list[pred_index]

# ✅ ROUTES
@app.route('/')
def index():
    return render_template('index1.html')

# 🟢 FORM SUBMIT → Store user data + Predict disease + Redirect to dashboard
@app.route('/submit_form', methods=['POST'])
def submit_form():
    try:
        name = request.form.get('name')
        age = request.form.get('age')
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        family_history = request.form.get('family_history')
        symptoms_str = request.form.get('symptoms')

        user_symptoms = [s.strip() for s in symptoms_str.split(',') if s.strip()]
        predicted_disease = get_predicted_value(user_symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)

        user_data = {
            "name": name,
            "age": age,
            "height": height,
            "weight": weight,
            "family_history": family_history,
            "symptoms": user_symptoms,
            "predicted_disease": predicted_disease
        }
        user_collection.insert_one(user_data)

        return render_template(
            'dashboard.html',
            name=name,
            predicted_disease=predicted_disease,
            dis_des=desc,
            my_precautions=pre,
            medications=med,
            my_diet=die,
            workout=wrkout
        )

    except Exception as e:
        print("Error:", e)
        return render_template('dashboard.html', disease="Error: " + str(e))


# ... (rest of app.py remains the same)
# 🆕 NEW ROUTE: Render dashboard.html with latest user data
# 🆕 ADD THIS EXACTLY HERE (no extra spaces/tabs before @app.route)
@app.route('/dashboard')
def dashboard():
    # Fetch the latest user data from MongoDB
    user = user_collection.find_one(sort=[('_id', -1)])  # Get the most recent entry
    if not user:
        return "No user data found. Please submit the form first.", 404

    return render_template(
        'dashboard.html',
        name=user['name'],
        predicted_disease=user['predicted_disease'],
        symptoms=user['symptoms'],
        dis_des=user.get('description', 'No description available'),
        my_precautions=user.get('precautions', []),
        medications=user.get('medicines', []),
        my_diet=user.get('diets', []),
        workout=user.get('workout', [])
    )

    


   

# 🟡 AJAX Route for Dashboard Info
@app.route('/get_info')
def get_info():
    disease = request.args.get('disease')
    info_type = request.args.get('type')

    desc, pre, med, die, wrk = helper(disease)
    info_map = {
        "description": desc,
        "precautions": pre,
        "medicine": med,
        "diet": die,
        "workout": wrk
    }

    return jsonify({"info": info_map.get(info_type, "No information found.")})

# 🔵 Generate & Display Health Report
from datetime import datetime

from datetime import datetime

@app.route('/generate_report')
def generate_report():
    user = user_collection.find_one(sort=[('_id', -1)])  # latest record
    if not user:
        return "No user data found"

    # calculate BMI
    height = float(user.get('height', 0))
    weight = float(user.get('weight', 0))
    bmi = round(weight / ((height / 100) ** 2), 2)
    bmi_status = (
        "Underweight" if bmi < 18.5 else
        "Normal" if bmi < 24.9 else
        "Overweight" if bmi < 29.9 else "Obese"
    )

    current_time = datetime.now().strftime("%d-%m-%Y at %I:%M %p")

    return render_template(
        'report.html',
        name=user.get('name'),
        age=user.get('age'),
        weight=weight,
        height=height,
        address=user.get('address'),
        predicted_disease=user.get('predicted_disease'),
        dis_des=user.get('description', 'No description available'),
        my_precautions=user.get('precautions', []),
        medications=user.get('medicines', []),
        my_diet=user.get('diets', []),
        workout=user.get('workout', []),
        bmi=f"{bmi} ({bmi_status})",
        due_date=current_time
    )


# 🟢 JSON API Route for Predict (used by index1.html) - UPDATED: Redirect after storage
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🧾 Collect user input from form
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        symptoms = request.form.get('symptoms')
        family_history = request.form.get('family_history')
        address = request.form.get('address')

        # 🔹 Convert symptoms to list
        symptoms_list = [s.strip() for s in symptoms.split(',') if s.strip()]

        # 🔹 Predict disease
        predicted_disease = get_predicted_value(symptoms_list)

        # 🔹 Get full details (description, medicines, etc.)
        desc, precautions_list, medicines_list, diets_list, workout_list = helper(predicted_disease)

        # ✅ Store full record in MongoDB
        record = {
            "name": name,
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "symptoms": symptoms_list,
            "family_history": family_history,
            "address": address,
            "predicted_disease": predicted_disease,
            "description": desc,
            "precautions": precautions_list,
            "medicines": medicines_list,
            "diets": diets_list,
            "workout": workout_list
        }

        user_collection.insert_one(record)
        print("✅ Data stored successfully:", record)

        # 🔁 Redirect to dashboard after saving
        return redirect(url_for('dashboard'))

    except Exception as e:
        print(f"❌ Error in /predict route: {e}")
        return "Error processing prediction", 500



if __name__ == '__main__':
    app.run(debug=True)
