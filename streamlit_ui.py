import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
# Fix the unicode emoji representation in the title
st.title('🦟 Malaria Severity Predictor')

# Initialize request counter in session state
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

@st.cache_resource
def load_model():
    # Load pipeline and feature names
    return (joblib.load('model/pipeline.joblib'),
            joblib.load('model/features.joblib'))

pipeline, FEATURES = load_model()

# -----------------------------------------------------------------------------
# Sidebar: Batch Prediction & Feature Importance
# -----------------------------------------------------------------------------
st.sidebar.header('Batch Prediction')
uploaded_file = st.sidebar.file_uploader('Upload CSV for Batch Prediction\nThe columns should must match the required features', type=['csv'])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        batch_df = pd.read_csv(uploaded_file)
        
        # Ensure all required features are present
        X_batch = batch_df[FEATURES]
        
        # Make predictions
        batch_df['Severity_Prediction'] = pipeline.predict(X_batch)
        batch_df['Severity_Prediction'] = batch_df['Severity_Prediction'].map({1: 'Severe', 0: 'Not Severe'})
        
        st.sidebar.success('Batch Processing Complete!')
        
        # Provide download button for results
        st.sidebar.download_button(
            label='Download Results',
            data=batch_df.to_csv(index=False).encode('utf-8'),
            file_name='batch_predictions_results.csv',
            mime='text/csv'
        )
    except KeyError:
        st.sidebar.error("Error: Uploaded file is missing required feature columns.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")

st.sidebar.markdown("---")
st.sidebar.header('Model Feature Importances')
# Try extracting feature importances from the last step of the pipeline
try:
    model = pipeline[-1]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Ensure the length matches before attempting to map
        if len(importances) == len(FEATURES):
            importance_series = pd.Series(importances, index=FEATURES)
            top_5_features = importance_series.sort_values(ascending=False).head(5)
            st.sidebar.bar_chart(top_5_features)
        else:
            st.sidebar.info("Feature lengths do not match pipeline configuration.")
    else:
        st.sidebar.info("The loaded model does not expose feature importances.")
except Exception as e:
    st.sidebar.warning(f"Could not load feature importances: {e}")


# -----------------------------------------------------------------------------
# Main Application UI: Patient Input
# -----------------------------------------------------------------------------
st.subheader("Patient Clinical Data")

# Form inputs for standard prediction
age = st.number_input('Age (years)', min_value=0, max_value=120, value=25)
sex = 1 if st.selectbox('Sex', ['Female', 'Male']) == 'Male' else 0

st.markdown("### Symptoms (Check all that apply)")
# Symptom Checkboxes
col1, col2, col3 = st.columns(3)
with col1:
    fever = 1 if st.checkbox('Fever') else 0
    cold = 1 if st.checkbox('Cold') else 0
    rigor = 1 if st.checkbox('Rigor') else 0
    fatigue = 1 if st.checkbox('Fatigue') else 0
    headace = 1 if st.checkbox('Headache') else 0
with col2:
    bitter_tongue = 1 if st.checkbox('Bitter Tongue') else 0
    vomitting = 1 if st.checkbox('Vomiting') else 0
    diarrhea = 1 if st.checkbox('Diarrhea') else 0
    Convulsion = 1 if st.checkbox('Convulsion') else 0
    Anemia = 1 if st.checkbox('Anemia') else 0
with col3:
    jundice = 1 if st.checkbox('Jundice') else 0
    cocacola_urine = 1 if st.checkbox('Cocacola Urine') else 0
    hypoglycemia = 1 if st.checkbox('Hypoglycemia') else 0
    prostraction = 1 if st.checkbox('Prostraction') else 0
    hyperpyrexia = 1 if st.checkbox('Hyperpyrexia') else 0

st.markdown("---")

# -----------------------------------------------------------------------------
# Prediction Execution & Explanations
# -----------------------------------------------------------------------------
if st.button('Run Prediction'):
    # Increment Request Counter
    st.session_state.prediction_count += 1
    
    # Construct input dataframe
    row = {'age': age, 'sex': sex, 'fever': fever, 'cold': cold, 'rigor': rigor, 
           'fatigue': fatigue, 'headace': headace, 'bitter_tongue': bitter_tongue, 
           'vomitting': vomitting, 'diarrhea': diarrhea, 'Convulsion': Convulsion, 
           'Anemia': Anemia, 'jundice': jundice, 'cocacola_urine': cocacola_urine, 
           'hypoglycemia': hypoglycemia, 'prostraction': prostraction, 'hyperpyrexia': hyperpyrexia}
    
    X = pd.DataFrame([row], columns=FEATURES)
    
    # Run Inference
    pred = int(pipeline.predict(X)[0])
    prob = float(pipeline.predict_proba(X)[0][1])
    
    # Determine Risk Label & Color
    if prob >= 0.7:
        risk, color = 'HIGH', 'red'
    elif prob >= 0.4:
        risk, color = 'MEDIUM', 'orange'
    else:
        risk, color = 'LOW', 'green'

    # Display Metrics and Outcome
    colA, colB = st.columns(2)
    with colA:
        st.success(f'Prediction: **{"Severe" if pred==1 else "Not Severe"}**')
        st.metric('Probability of Severe Malaria', f'{prob:.1%}')
    with colB:
        st.info(f"Predictions made this session: **{st.session_state.prediction_count}**")
        # Color-coded risk badge using HTML
        st.markdown(
            f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; text-align: center; font-size: 20px; font-weight: bold;">'
            f'Risk level: {risk}'
            f'</div>', 
            unsafe_allow_html=True
        )

    # SHAP Explanation Panel
    st.markdown("---")
    st.subheader('Explanation of Prediction (SHAP)')
    try:
        # Wrap the predict pipeline in a SHAP explainer
        # Using the agnostic Explainer (handles pipelines natively by passing the predict function)
        explainer = shap.Explainer(pipeline.predict, X)
        shap_values = explainer(X)
        shap_values.feature_names = FEATURES
        
        # Plotting the waterfall
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation. Note: Ensure your pipeline structure is fully compatible with SHAP. Details: {e}")