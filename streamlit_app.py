# streamlit_app.py — Core pattern
import streamlit as st
import joblib, pandas as pd

@st.cache_resource # load once, reuse on every rerun

def load_model():
    return (joblib.load('model/pipeline.joblib'),
            joblib.load('model/features.joblib'))

pipeline, FEATURES = load_model()

st.title('U0001F99F Malaria Severity Predictor')

age = st.number_input('Age (years)', min_value=0, max_value=120, value=25)
sex = 1 if st.selectbox('Sex', ['Female', 'Male']) == 'Male' else 0
fever = 1 if st.checkbox('Fever') else 0
cold = 1 if st.checkbox('Cold') else 0
rigor = 1 if st.checkbox('Rigor') else 0
fatigue = 1 if st.checkbox('Fatigue') else 0
headace = 1 if st.checkbox('Headache') else 0
bitter_tongue = 1 if st.checkbox('Bitter Tongue') else 0
vomitting = 1 if st.checkbox('Vomiting') else 0
diarrhea = 1 if st.checkbox('Diarrhea') else 0
Convulsion = 1 if st.checkbox('Convulsion') else 0
Anemia = 1 if st.checkbox('Anemia') else 0
jundice = 1 if st.checkbox('Jundice') else 0
cocacola_urine = 1 if st.checkbox('Cocacola Urine') else 0
hypoglycemia = 1 if st.checkbox('Hypoglycemia') else 0
prostraction = 1 if st.checkbox('Prostraction') else 0
hyperpyrexia = 1 if st.checkbox('Hyperpyrexia') else 0


if st.button('Run Prediction'):
    row = {'age': age, 'sex': sex, 'fever': fever, 'cold': cold, 'rigor': rigor, 'fatigue': fatigue, 'headace': headace,
           'bitter_tongue': bitter_tongue, 'vomitting': vomitting, 'diarrhea': diarrhea, 'Convulsion': Convulsion, 'Anemia': Anemia,
           'jundice': jundice, 'cocacola_urine': cocacola_urine, 'hypoglycemia': hypoglycemia, 'prostraction': prostraction, 'hyperpyrexia': hyperpyrexia}
    
    X = pd.DataFrame([row], columns=FEATURES)
    pred = int(pipeline.predict(X)[0])
    prob = float(pipeline.predict_proba(X)[0][1])
    risk = 'HIGH' if prob>=0.7 else 'MEDIUM' if prob>=0.4 else 'LOW'

    st.success(f'Prediction: {"Severe" if pred==1 else "Not Severe"}')
    st.metric('Probability of Severe Malaria', f'{prob:.1%}')
    st.write(f'Risk level: {risk}')