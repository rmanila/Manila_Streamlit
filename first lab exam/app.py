import streamlit as st
import joblib
import numpy as np



model = joblib.load("heart_risk_model.pkl")
encoders = joblib.load("label_encoders.pkl")


st.title("Heart Disease Risk Predictor")
st.write("Based on clinical details (BMI and other 7 fields)")

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
physical_health = st.number_input("Physical Health (0–30)", 0.0, 30.0, 5.0)
mental_health = st.number_input("Mental Health (0–30)", 0.0, 30.0, 5.0)
sleep_time = st.number_input("Sleep Time (hours/day)", 0.0, 24.0, 7.0)

smoking = st.selectbox("Do you smoke?", list(encoders['Smoking'].classes_))
sex = st.selectbox("Sex", list(encoders['Sex'].classes_))
age = st.selectbox("Age Category", list(encoders['AgeCategory'].classes_))
gen_health = st.selectbox("General Health", list(encoders['GenHealth'].classes_))

def encode(encoder, value, name):
    try:
        return encoder.transform([value])[0]
    except:
        st.error(f"Invalid input for {name}")
        st.stop()

encoded_input = [
    bmi,
    physical_health,
    mental_health,
    sleep_time,
    encode(encoders['Smoking'], smoking, "Smoking"),
    encode(encoders['Sex'], sex, "Sex"),
    encode(encoders['AgeCategory'], age, "AgeCategory"),
    encode(encoders['GenHealth'], gen_health, "GenHealth")
]


if st.button("Predict"):
    pred = model.predict([encoded_input])[0]
    prob = model.predict_proba([encoded_input])[0]

    label = "At Risk" if pred == 1 else "Not At Risk"
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {np.max(prob) * 100:.2f}%")
