"""
Diabetes Prediction Streamlit App
Interactive web application for predicting diabetes risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

# Handle pyarrow compatibility issues
# Convert DataFrames to compatible types to avoid pyarrow serialization issues
def safe_dataframe_display(df, **kwargs):
    """Safely display dataframe, with fallback to table if pyarrow issues occur"""
    try:
        # Ensure data types are compatible - convert to native Python types
        df_copy = df.copy()
        # Convert all columns to native Python types to avoid pyarrow issues
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].astype(str)
            elif df_copy[col].dtype in ['int64', 'int32']:
                df_copy[col] = df_copy[col].astype('int32')
            elif df_copy[col].dtype in ['float64', 'float32']:
                df_copy[col] = df_copy[col].astype('float32')
        # Convert deprecated use_container_width to width parameter
        if 'use_container_width' in kwargs:
            if kwargs['use_container_width']:
                kwargs['width'] = 'stretch'
            else:
                kwargs['width'] = 'content'
            del kwargs['use_container_width']
        # Try to display as dataframe
        return st.dataframe(df_copy, **kwargs)
    except Exception as e:
        # Fallback to table display (doesn't require pyarrow)
        return st.table(df)

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Diabetes Prediction App")
st.markdown("""
This application uses a machine learning model (RandomForestClassifier) to predict the likelihood of diabetes 
based on several health metrics. The model was trained on the Pima Indians Diabetes Dataset.

**How it works:**
1. Enter your health metrics in the sidebar
2. Click the 'Predict' button
3. The model will analyze your inputs and provide a prediction with probability scores

**Note:** This is a prediction tool for educational purposes and should not replace professional medical advice.
""")

# Load model, scaler, and metadata
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and metadata"""
    try:
        model = joblib.load('models/diabetes_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except FileNotFoundError:
        st.error("Model files not found! Please run 'python src/model.py' first to train the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler, metadata = load_model()

# Sidebar for user inputs
st.sidebar.header("📊 Enter Health Metrics")

# Define feature ranges based on the dataset characteristics
pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
    help="Number of times pregnant"
)

glucose = st.sidebar.number_input(
    "Glucose (mg/dL)",
    min_value=0.0,
    max_value=300.0,
    value=100.0,
    step=0.1,
    format="%.1f",
    help="Plasma glucose concentration (2 hours in oral glucose tolerance test)"
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0.0,
    max_value=150.0,
    value=70.0,
    step=0.1,
    format="%.1f",
    help="Diastolic blood pressure"
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness (mm)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.1,
    format="%.1f",
    help="Triceps skin fold thickness"
)

insulin = st.sidebar.number_input(
    "Insulin (mu U/ml)",
    min_value=0.0,
    max_value=1000.0,
    value=80.0,
    step=0.1,
    format="%.1f",
    help="2-Hour serum insulin"
)

bmi = st.sidebar.number_input(
    "BMI (kg/m²)",
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    step=0.1,
    format="%.1f",
    help="Body Mass Index"
)

diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5,
    step=0.01,
    format="%.3f",
    help="Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)"
)

age = st.sidebar.number_input(
    "Age (years)",
    min_value=0,
    max_value=120,
    value=30,
    step=1,
    help="Age in years"
)

# Prediction button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)  # Button doesn't support width parameter yet

# Main content area
if predict_button:
    # Check for potentially invalid 0 values
    zero_features = []
    if glucose == 0: zero_features.append("Glucose")
    if blood_pressure == 0: zero_features.append("Blood Pressure")
    if skin_thickness == 0: zero_features.append("Skin Thickness")
    if insulin == 0: zero_features.append("Insulin")
    if bmi == 0: zero_features.append("BMI")
    
    if zero_features:
        st.warning(f"⚠️ **Warning**: You entered 0 for: **{', '.join(zero_features)}**. In a medical context, a value of 0 for these metrics is usually impossible and indicates missing data. This may affect the prediction.")

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Display input summary
    st.subheader("📋 Your Input Summary")
    safe_dataframe_display(input_data, use_container_width=True)
    
    # Preprocess: Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Use optimal threshold from metadata if available
    threshold = metadata.get('optimal_threshold', 0.5)
    diabetic_prob = prediction_proba[1]
    
    # Display results
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Add Borderline confidence warning (e.g., probability within 10% of threshold)
        if threshold - 0.10 <= diabetic_prob <= threshold + 0.10:
            st.warning("### Prediction: **Borderline** ⚠️")
            st.info("The model is moderately uncertain about the prediction. Please consult a healthcare professional for clinical evaluation.")
        elif diabetic_prob >= threshold:
            st.error("### Prediction: **Diabetic** 🛑")
            st.warning("The model predicts a high likelihood of diabetes. Please consult with a healthcare professional.")
        else:
            st.success("### Prediction: **Not Diabetic** ✅")
            st.info("The model predicts a low likelihood of diabetes. Continue maintaining a healthy lifestyle.")
    
    with col2:
        st.metric("Probability (Not Diabetic)", f"{prediction_proba[0]*100:.2f}%")
        st.metric("Probability (Diabetic)", f"{prediction_proba[1]*100:.2f}%")
        st.caption(f"*Decision Threshold Used: {threshold*100:.1f}%*")
    
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Probability visualization
        st.subheader("📊 Probabilities")
        prob_df = pd.DataFrame({
            'Outcome': ['Not Diabetic', 'Diabetic'],
            'Probability': [float(prediction_proba[0]*100), float(prediction_proba[1]*100)]
        })
        try:
            prob_df['Probability'] = prob_df['Probability'].astype('float32')
            st.bar_chart(prob_df.set_index('Outcome'))
        except Exception:
            st.write(f"**Not Diabetic:** {prediction_proba[0]*100:.2f}%")
            st.write(f"**Diabetic:** {prediction_proba[1]*100:.2f}%")
            
    with col_chart2:
        # Feature Importance Visualization
        if 'feature_importance' in metadata:
            st.subheader("💡 Feature Importance")
            fi = metadata['feature_importance']
            fi_df = pd.DataFrame({
                'Feature': list(fi.keys()),
                'Importance': list(fi.values())
            }).sort_values('Importance', ascending=True)
            
            try:
                st.bar_chart(fi_df.set_index('Feature'), horizontal=True)
            except Exception:
                # Fallback for older streamlit versions without horizontal support
                st.bar_chart(fi_df.set_index('Feature'))
                
            st.caption("Shows which metrics generally have the most impact on predictions overall.")

else:
    # Default message when app loads
    st.info("👈 Please enter your health metrics in the sidebar and click 'Predict' to get started.")
    
    # Display feature descriptions
    st.markdown("---")
    st.subheader("📖 Feature Descriptions")
    
    feature_info = pd.DataFrame({
        'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                   'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
        'Description': [
            'Number of times the patient has been pregnant.',
            'Plasma glucose concentration after a 2-hour oral glucose tolerance test. Measures how well the body processes sugar.',
            'Diastolic blood pressure (mm Hg). The pressure in the arteries when the heart rests between beats.',
            'Triceps skin fold thickness (mm). Used to estimate body fat percentage.',
            '2-Hour serum insulin (mu U/ml). Measures the amount of insulin in the blood.',
            'Body Mass Index. A measure of body fat based on weight and height (kg/m²).',
            'A genetic score that estimates the genetic risk of diabetes based on family history. Higher values indicate higher risk.',
            'Patient age in years.'
        ]
    })
    
    safe_dataframe_display(feature_info, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</small>
</div>
""", unsafe_allow_html=True)

