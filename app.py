import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction Tool",
    page_icon="🩺",
    layout="wide"
)

MODEL_PATH = 'diabetes_model_with_threshold.pkl'

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please run 'diabetes_classification.py' first.")
        return None
    return joblib.load(MODEL_PATH)

bundle = load_model()

# --- Main App ---
def main():
    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown("""
    This tool uses a machine learning model to estimate the likelihood of diabetes based on diagnostic measures.
    **Note:** Zeros in Insulin, Skin Thickness, BMI, etc., are automatically treated as "missing values" by the model.
    """)

    if bundle is None:
        return

    # Extract pipeline and threshold from the bundle
    pipeline = bundle['pipeline']
    threshold = bundle['threshold']

    # --- Sidebar Inputs ---
    st.sidebar.header("Patient Data")
    
    def user_input_features():
        pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
        glucose = st.sidebar.number_input("Glucose Level (mg/dL)", 0, 300, 120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test.")
        blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", 0, 150, 70, help="Diastolic blood pressure.")
        skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness.")
        insulin = st.sidebar.number_input("Insulin (mu U/ml)", 0, 900, 79, help="2-Hour serum insulin.")
        bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 32.0, format="%.1f", help="Body mass index (weight in kg/(height in m)^2).")
        dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, format="%.3f", help="Diabetes pedigree function.")
        age = st.sidebar.slider("Age (years)", 21, 100, 33)

        data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        return pd.DataFrame([data])

    input_df = user_input_features()

    # --- Display Input Data ---
    with st.expander("View Input Data Summary"):
        st.dataframe(input_df)

    # --- Prediction ---
    if st.button("Predict Risk", type="primary"):
        with st.spinner('Calculating...'):
            # Get probability
            probability = pipeline.predict_proba(input_df)[0, 1]
            
            # Determine class based on saved threshold
            prediction = 1 if probability >= threshold else 0
            
            # --- Results Display ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction")
                if prediction == 1:
                    st.error("🛑 High Risk of Diabetes")
                else:
                    st.success("✅ Low Risk of Diabetes")
            
            with col2:
                st.subheader("Probability Score")
                st.metric(label="Risk Probability", value=f"{probability:.2%}", delta=f"Threshold: {threshold:.2f}")
                
                # Visual Gauge using Progress Bar
                st.progress(probability)
                
                if prediction == 1:
                    st.caption(f"The model predicts **Positive** because the probability ({probability:.2f}) is higher than the decision threshold ({threshold}).")
                else:
                    st.caption(f"The model predicts **Negative** because the probability ({probability:.2f}) is lower than the decision threshold ({threshold}).")

if __name__ == "__main__":
    main()