import os
import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="NovaGen Health Risk Classifier", layout="centered")
st.title("🌲 NovaGen Health Risk Diagnostic Engine")
st.write("Enter patient metrics below to compute real-time health risk stratification.")

# 2. Load Pre-trained Artifacts Safely
@st.cache_resource
def load_model_artifacts():
    model_path = os.path.join("models", "random_forest_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("⚠️ Model artifacts missing! Please run 'python src/train.py' first.")
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_artifacts()

# 3. Build UI Input Fields
st.subheader("📋 Patient Vitals & Clinical Metrics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
    blood_pressure = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=90)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=72)

with col2:
    sleep = st.slider("Daily Sleep Hours", 3.0, 12.0, 7.0)
    exercise = st.slider("Weekly Exercise Hours", 0.0, 30.0, 3.0)
    water = st.slider("Daily Water Intake (L)", 0.5, 5.0, 2.0)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Active Smoker"])
    alcohol = st.selectbox("Alcohol Consumption", ["None/Occasional", "Regular"])

# 4. Run Prediction
if st.button("🚀 Calculate Risk Stratification", type="primary"):
    if model and scaler:
        raw_payload = {
            'Age': age, 'BMI': bmi, 'Blood_Pressure': blood_pressure, 
            'Cholesterol': cholesterol, 'Glucose_Level': glucose, 'Heart_Rate': heart_rate,
            'Sleep_Hours': sleep, 'Exercise_Hours': exercise, 'Water_Intake': water, 
            'Stress_Level': stress,
            'Smoking': 1 if smoking == "Active Smoker" else 0,
            'Alcohol': 1 if alcohol == "Regular" else 0,
            # Static columns to match the trained data shape
            'Diet': 1, 'MentalHealth': 5, 'PhysicalActivity': 3, 'MedicalHistory': 0, 'Allergies': 0,
            'Diet_Type__Vegan': 0, 'Diet_Type__Vegetarian': 0,
            'Blood_Group_AB': 0, 'Blood_Group_B': 0, 'Blood_Group_O': 1
        }
        
        input_df = pd.DataFrame([raw_payload])
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **High Risk Detected** (Confidence: {probabilities[1]*100:.1f}%)")
        else:
            st.success(f"✅ **Low/Normal Risk Stratification** (Confidence: {probabilities[0]*100:.1f}%)")