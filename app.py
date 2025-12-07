"""
Software Defect Type Prediction - Streamlit App
Multi-Label Classification from Bug Report Text
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Defect Prediction", page_icon="🐛", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 2.5rem; font-weight: bold; color: #e74c3c; text-align: center; 
padding: 1rem 0; border-bottom: 3px solid #e74c3c; margin-bottom: 2rem;}
.prediction-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
color: white; padding: 2rem; border-radius: 15px; margin: 1.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.2);}
.defect-label {background-color: #e74c3c; color: white; padding: 0.5rem 1rem; 
border-radius: 20px; display: inline-block; margin: 0.3rem; font-weight: 600;}
.no-defect {background-color: #27ae60; color: white; padding: 0.5rem 1rem; 
border-radius: 20px; display: inline-block; margin: 0.3rem; font-weight: 600;}

/* Styled radio buttons */
div[role="radiogroup"] {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}
div[role="radiogroup"] label {
    background-color: white;
    border: 2px solid #dfe1e5;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 150px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
div[role="radiogroup"] label:hover {
    border-color: #e74c3c;
    box-shadow: 0 2px 8px rgba(231,76,60,0.15);
    transform: translateY(-1px);
}
div[role="radiogroup"] label:has(input:checked) {
    background-color: #fff5f5;
    border-color: #e74c3c;
    border-width: 2px;
    color: #e74c3c;
    font-weight: 600;
    box-shadow: 0 2px 12px rgba(231,76,60,0.25);
}
div[role="radiogroup"] input[type="radio"] {
    margin-right: 0.5rem;
}

/* Mobile responsive */
@media (max-width: 768px) {
    div[role="radiogroup"] {
        flex-direction: column;
        gap: 0.5rem;
    }
    div[role="radiogroup"] label {
        width: 100%;
        min-width: unset;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models and vectorizer"""
    models = {}
    vectorizer = None
    
    # Load vectorizer
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except Exception as e:
        st.error(f"Failed to load vectorizer: {str(e)}")
        st.info("Run cells 13-16 in the notebook to export models")
        return {}, None
    
    # Load models
    try:
        models['Logistic Regression'] = joblib.load('logistic_regression_model.pkl')
    except: pass
    
    try:
        models['SVM'] = joblib.load('svm_model.pkl')
    except: pass
    
    try:
        models['Perceptron'] = joblib.load('perceptron_model.pkl')
    except: pass
    
    try:
        models['Deep Neural Network'] = keras.models.load_model('dnn_model.h5', compile=False)
    except: pass
    
    return models, vectorizer

def predict(model, features, model_name):
    """Make predictions"""
    if model_name == 'Deep Neural Network':
        probs = model.predict(features.toarray(), verbose=0)[0]
        preds = (probs >= 0.5).astype(int)
    else:
        preds = model.predict(features)[0]
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[0]
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(features)[0]
            probs = 1 / (1 + np.exp(-scores))
        else:
            probs = preds.astype(float)
    return preds, probs

# Main app
st.markdown('<div class="main-header">🐛 Software Defect Type Prediction</div>', unsafe_allow_html=True)

models, vectorizer = load_models()

if not models:
    st.error("No models found. Run the notebook to train and export models.")
    st.stop()

# Main content
st.info("Enter a bug report description to predict defect types")

input_method = st.radio("Input Method:", ["Text Entry", "CSV Upload"], horizontal=True)

bug_report = None

if input_method == "Text Entry":
    bug_report = st.text_area(
        "Bug Report Description",
        height=200,
        placeholder="Example: Critical bug in authentication causing system crash..."
    )
    
    with st.expander("💡 Example Reports"):
        st.code("Critical authentication failure causing complete system lockout", language=None)
        st.code("Add dark mode support to improve user experience", language=None)
        st.code("Update React version to address security vulnerabilities", language=None)

else:
    uploaded = st.file_uploader("Upload CSV with 'report' column", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)
        if 'report' in df.columns:
            bug_report = df['report'].iloc[0]
            st.success(f"Loaded: {bug_report[:100]}...")

# Model selection
st.markdown("### ⚙️ Model Selection")
model_choice = st.radio("Choose prediction model:", list(models.keys()), horizontal=True, label_visibility="collapsed")

# Prediction
if bug_report and st.button("🚀 Predict Defect Types", use_container_width=True):
    with st.spinner(f"Analyzing with {model_choice}..."):
        features = vectorizer.transform([bug_report])
        preds, probs = predict(models[model_choice], features, model_choice)
        
        labels = ['Blocker', 'Regression', 'Bug', 'Documentation', 'Enhancement', 'Task', 'Dependency Upgrade']
        
        # Results
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Results")
        
        num_detected = np.sum(preds)
        if num_detected == 0:
            st.markdown('<div class="no-defect">✅ No Defect Types Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{num_detected} Type(s) Detected:**")
            html = ""
            for pred, label in zip(preds, labels):
                if pred == 1:
                    html += f'<div class="defect-label">🐛 {label}</div>'
            st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confidence scores
        st.markdown("### 📊 Confidence Scores")
        df_results = pd.DataFrame({
            'Defect Type': labels,
            'Prediction': ['✓ Detected' if p == 1 else '✗ Not Detected' for p in preds],
            'Confidence': [f"{prob:.1%}" for prob in probs]
        })
        
        def highlight(row):
            return ['background-color: #ffe6e6' if row['Prediction'] == '✓ Detected' 
                    else 'background-color: #e6f7e6'] * len(row)
        
        st.dataframe(df_results.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)
        
        # Bar chart
        st.bar_chart(pd.DataFrame({'Probability': probs}, index=labels))
        
        # Download
        csv = df_results.to_csv(index=False)
        st.download_button("💾 Download Results", csv, "predictions.csv", "text/csv", use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #7f8c8d;'>Built with Streamlit • Powered by ML</div>", 
            unsafe_allow_html=True)
