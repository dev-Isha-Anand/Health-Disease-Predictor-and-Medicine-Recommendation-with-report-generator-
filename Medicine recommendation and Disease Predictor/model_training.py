import re
import difflib
import  pandas as pd
dataset = pd.read_csv("datasets/Training.csv")
print(dataset)
print(dataset.shape)
# print the unique disease name
print(dataset['prognosis'].unique())
print(len(dataset['prognosis'].unique()))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# ecoding prognonsis
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Create a dictionary to store models
models = {
    'SVC': SVC(kernel='linear'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'MultinomialNB': MultinomialNB()
}

# Loop through the models, train, test, and print results
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"{model_name} Confusion Matrix:")
    print(np.array2string(cm, separator=', '))

    print("\n" + "="*40 + "\n")
# selecting svc
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)
ypred = svc.predict(X_test)
accuracy_score(y_test,ypred)
# save svc
import pickle
pickle.dump(svc,open('svc.pkl','wb'))
# load model
svc = pickle.load(open('svc.pkl','rb'))
# test 1:
print("predicted disease :",svc.predict(X_test.iloc[0].values.reshape(1,-1)))
print("Actual Disease :", y_test[0])
# test 2:
print("predicted disease :",svc.predict(X_test.iloc[100].values.reshape(1,-1)))
print("Actual Disease :", y_test[100])
# Predict on test set
y_pred = svc.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/disease_specific_yoga_workout_all_42.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/workout_df.csv")  # Now this is the original diets.csv
#============================================================
# custom and helping functions
#==========================helper functions================
def helper(dis):
    # Determine the column name for disease (handle both 'Disease' and 'disease')
    disease_col = 'Disease' if 'Disease' in description.columns else 'disease'

    desc = description[description[disease_col] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions[disease_col] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications[disease_col] == dis]['Medication']
    med = [m for m in med.values]

    # For diets (now workout_df.csv), use 'workout' column instead of 'Diet'
    die = diets[diets[disease_col] == dis]['workout']
    die = [d for d in die.values]

    # For workout, check for 'disease' or 'Disease'
    if 'disease' in workout.columns:
        wrkout = workout[workout['disease'] == dis]['workout']
    else:
        wrkout = workout[workout[disease_col] == dis]['workout'] if disease_col in workout.columns else []

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

_symptom_keys = list(symptoms_dict.keys())

def normalize_symptom(s: str) -> str:
    """
    Normalize a user-typed symptom to a canonical form suitable for matching
    - lowercases
    - strips leading/trailing spaces
    - replaces spaces and hyphens with underscores
    - removes characters other than letters, numbers and underscore
    - collapses multiple underscores
    """
    if s is None:
        return ""
    s = s.strip().lower()
    # Replace spaces and hyphens with underscore
    s = re.sub(r'[\s\-]+', '_', s)
    # Remove characters other than letters, numbers and underscore
    s = re.sub(r'[^a-z0-9_]', '', s)
    # collapse multiple underscores
    s = re.sub(r'_+', '_', s)
    return s

def match_symptom_key(user_symptom: str, cutoff=0.7):
    """
    Try to map user_symptom (raw) to a key in symptoms_dict.
    1) exact normalized match
    2) try common manual fixes (if needed)
    3) fuzzy match using difflib.get_close_matches (returns best match or None)
    """
    if not user_symptom:
        return None

    norm = normalize_symptom(user_symptom)

    # Quick exact match
    if norm in symptoms_dict:
        return norm

    # small manual mapping examples (extend as you learn common user inputs)
    manual_map = {
        'skinrash': 'skin_rash',
        'spottingurination': 'spotting_ urination',  # handle your dataset odd spacing
        'coldhandsandfeets': 'cold_hands_and_feets',
        'yellowishskin': 'yellowish_skin',
        'highfever': 'high_fever',
        'mildfever': 'mild_fever',
        'stomachpain': 'stomach_pain'
    }
    if norm in manual_map:
        return manual_map[norm]

    # If not exact, try fuzzy matching on original keys
    # Use difflib to find close matches
    # create candidate list of keys with underscores removed also to help match
    keys_for_match = _symptom_keys + [k.replace('_', '') for k in _symptom_keys]
    matches = difflib.get_close_matches(norm, keys_for_match, n=1, cutoff=cutoff)
    if matches:
        candidate = matches[0]
        # if we matched a key without underscores, map back to original key
        if candidate in symptoms_dict:
            return candidate
        else:
            # try to re-insert underscores by comparing with original keys
            # find the original key whose underscores-removed form equals candidate
            for k in _symptom_keys:
                if k.replace('_', '') == candidate:
                    return k
    # nothing found
    return None

# Model Prediction function
def get_predicted_value(patient_symptoms):
    """
    patient_symptoms: list of raw strings from user
    returns: predicted disease name (string) OR raises ValueError if no valid symptom
    """
    # Initialize input vector of zeros
    input_vector = np.zeros(len(symptoms_dict))

    recognized = []   # list of matched symptom keys
    unrecognized = [] # keep track to inform/log

    for raw in patient_symptoms:
        matched_key = match_symptom_key(raw)
        if matched_key:
            recognized.append(matched_key)
            input_vector[symptoms_dict[matched_key]] = 1
        else:
            unrecognized.append(raw)

    if unrecognized:
        # print warnings — in a real app log these rather than printing
        print("Warning: these symptoms were not recognized and will be ignored:", unrecognized)

    if not recognized:
        # no valid symptom -> safe fallback
        raise ValueError("No recognizable symptoms provided. Please check spelling or enter different symptoms.")

    # Predict using the trained model (fix: use DataFrame to match feature names)
    input_df = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    pred_index = svc.predict(input_df)[0]
    return diseases_list[pred_index]
# Test 1
# Split the user's input into a list of symptoms (assuming they are comma-separated) # itching,skin_rash,nodal_skin_eruptions
symptoms = input("Enter your symptoms (comma-separated): ")
# split by comma and strip
user_symptoms = [s.strip() for s in symptoms.split(',') if s.strip() != '']
# Remove any extra characters, keep raw entries (we use normalization inside matching)
# user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]   # optional now

try:
    predicted_disease = get_predicted_value(user_symptoms)
except ValueError as e:
    print("Error:", e)
    # stop gracefully - you can return or exit depending on how you run this script
    raise SystemExit(1)

# Now call helper with predicted_disease (unchanged flow below)
desc, pre, med, die, wrkout = helper(predicted_disease)

print("=================predicted disease============")
print(predicted_disease)
print("=================description==================")
print(desc)
print("=================precautions==================")
i = 1
for p_i in pre[0]:
    print(f"{i}. {p_i}")
    i += 1

print("=================medications==================")
i = 1
import ast
for m in med:
    if isinstance(m, str):
        med_list = ast.literal_eval(m)  # Safely parse the string list
        for item in med_list:
            print(f"{i}. {item}")
            i += 1

print("=================workout==================")
i = 1
for w_i in wrkout:
    print(f"{i}. {w_i}")
    i += 1

print("=================diets==================")
i = 1
for d_i in die:
    print(f"{i}. {d_i}")
    i += 1
