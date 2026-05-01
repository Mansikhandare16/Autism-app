import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Autism Prediction", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

st.title("Autism Spectrum Prediction")
st.markdown("Fill in the details below to get a prediction.")

st.subheader("AQ-10 Questionnaire")
col1, col2 = st.columns(2)
with col1:
    A1 = st.selectbox("A1: Notices small sounds", [0, 1])
    A2 = st.selectbox("A2: Focuses on patterns", [0, 1])
    A3 = st.selectbox("A3: Multitasking is easy", [0, 1])
    A4 = st.selectbox("A4: Enjoys social chit-chat", [0, 1])
    A5 = st.selectbox("A5: Notices number plates", [0, 1])
with col2:
    A6 = st.selectbox("A6: Knows how to read feelings", [0, 1])
    A7 = st.selectbox("A7: Collects information", [0, 1])
    A8 = st.selectbox("A8: Finds it hard in social groups", [0, 1])
    A9 = st.selectbox("A9: Switches between tasks easily", [0, 1])
    A10 = st.selectbox("A10: Finds new situations easy", [0, 1])

result = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10
st.info(f"AQ Score: **{result}** / 10")

st.subheader("Personal Details")
col3, col4 = st.columns(2)
with col3:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ["f", "m"])
    jaundice = st.selectbox("Born with jaundice?", ["no", "yes"])
with col4:
    austim = st.selectbox("Family member with autism?", ["no", "yes"])
    used_app = st.selectbox("Used this app before?", ["no", "yes"])
    relation = st.selectbox("Who is filling this?", ["Self", "Others"])

ethnicity = st.selectbox("Ethnicity", [
    "White-European", "Asian", "Middle Eastern", "Black", "South Asian",
    "Hispanic", "Latino", "Pasifika", "Turkish", "Others"
])
country = st.selectbox("Country of residence", [
    "United States", "United Kingdom", "India", "Australia", "New Zealand",
    "Canada", "Jordan", "Saudi Arabia", "Egypt", "Others"
])

if st.button("Predict", use_container_width=True):
    raw = {
        "A1_Score": A1, "A2_Score": A2, "A3_Score": A3,
        "A4_Score": A4, "A5_Score": A5, "A6_Score": A6,
        "A7_Score": A7, "A8_Score": A8, "A9_Score": A9,
        "A10_Score": A10, "age": age, "result": result,
        "gender": gender, "ethnicity": ethnicity,
        "jaundice": jaundice, "austim": austim,
        "contry_of_res": country, "used_app_before": used_app,
        "relation": relation
    }
    df_input = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in df_input.columns:
            val = df_input[col].iloc[0]
            if val not in le.classes_:
                df_input[col] = le.transform([le.classes_[0]])
            else:
                df_input[col] = le.transform([val])

    # exact column order from training
    feature_cols = [
              'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
              'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
             'age', 'gender', 'ethnicity', 'jaundice', 'austim',
             'contry_of_res', 'used_app_before', 'result', 'relation'
    ]
    df_input = df_input[feature_cols]
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][pred]

    if pred == 1:
        st.error(f"🔴 Result: Autism traits detected  ({prob*100:.1f}% confidence)")
        st.markdown("*Please consult a qualified healthcare professional for a proper diagnosis.*")
    else:
        st.success(f"🟢 Result: No autism traits detected  ({prob*100:.1f}% confidence)")
        st.markdown("*This is a screening tool only, not a medical diagnosis.*")