import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config, ROOT_DIR
config = load_config()

# Configuration
st.set_page_config(page_title="Credit Card Fraud Detection Prototype", page_icon="💳", layout="wide")

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0b5394;
    color: white;
    width: 100%;
    font-weight: bold;
}
.fraud-alert {
    background-color: #ffcccc;
    padding: 20px;
    border-radius: 10px;
    border-left: 10px solid red;
    color: #990000;
}
.normal-alert {
    background-color: #d9ead3;
    padding: 20px;
    border-radius: 10px;
    border-left: 10px solid green;
    color: #274e13;
}
</style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS

@st.cache_resource
def load_models():
    models_dir = os.path.join(ROOT_DIR, config['paths']['models_dir'])
    if not os.path.exists(models_dir):
        st.error(f"No models found in {models_dir}. Please run main_pipeline.py first.")
        return {}
        
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    loaded_models = {}
    for f in model_files:
        raw_name = f.split('.')[0].upper()
        # Clean up names for UI
        if raw_name == "RF": model_name = "Random Forest"
        elif raw_name == "XGB": model_name = "XGBoost"
        elif raw_name == "LGB": model_name = "LightGBM"
        elif raw_name == "ENSEMBLE": model_name = "Stacking Ensemble"
        else: model_name = raw_name
        
        try:
            loaded_models[model_name] = joblib.load(os.path.join(models_dir, f))
        except Exception as e:
            st.error(f"Failed to load {f}: {e}")
            
    # Sắp xếp để Ensemble lên đầu nếu có
    sorted_models = {}
    if "Stacking Ensemble" in loaded_models:
        sorted_models["Stacking Ensemble"] = loaded_models.pop("Stacking Ensemble")
    sorted_models.update(loaded_models)
    
    return sorted_models

@st.cache_data
def load_test_data():
    test_data_path = os.path.join(ROOT_DIR, config['paths']['test_data'])
    test_label_path = os.path.join(ROOT_DIR, config['paths']['test_label'])
    if not os.path.exists(test_data_path):
        st.error(f"No test data found at {test_data_path}. Please run data_preprocessing.py first.")
        return None, None
    X_test = pd.read_csv(test_data_path)
    y_test = pd.read_csv(test_label_path)
    return X_test, y_test


# INITIALIZATION
models = load_models()
X_test, y_test = load_test_data()

st.title("💳 Credit Card Fraud Detection Prototype")
st.write("Machine Learning to detect real-time financial fraud.")

# UI COMPONENTS
if not models or X_test is None:
    st.warning("Please wait for the data to load or training to complete...")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Filter Settings")
    
    selected_model_name = st.selectbox(
        "Select Algorithm:",
        list(models.keys()),
        index=0
    )
    
    st.markdown("---")
    st.subheader("🎲 Transaction Simulation (Client)")
    
    if st.button("🔄 Get Random Transaction"):
        # Select random index
        st.session_state['current_idx'] = int(np.random.randint(0, len(X_test)))
        
    if st.button("⚠️ Get Fraud Transaction"):
        # Select a fraud case index
        fraud_indices = y_test[y_test['Class'] == 1].index
        st.session_state['current_idx'] = int(np.random.choice(fraud_indices))
        
    if st.button("✅ Get Normal Transaction"):
        normal_indices = y_test[y_test['Class'] == 0].index
        st.session_state['current_idx'] = int(np.random.choice(normal_indices))

with col2:
    st.header("🔍 Server")
    
    if 'current_idx' not in st.session_state:
        st.info("👈 Transaction information will appear here. Click the transaction button in the left menu!")
    else:
        idx = st.session_state['current_idx']
        transaction_data = X_test.iloc[[idx]]
        true_label = int(y_test.iloc[idx]['Class'])
        
        # Display Transaction Info simply
        st.subheader(f"Transaction ID #{idx}")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Transaction Amount ($)", f"{transaction_data['Amount'].values[0]:.2f}")
        with col_t2:
            st.metric("Transaction Time (seconds)", f"{transaction_data['Time'].values[0]:.0f}")
            
        with st.expander("Explore the variables after PCA (V1 - V28)"):
            st.dataframe(transaction_data.drop(columns=['Amount', 'Time']))
            
        st.markdown("---")
        
        st.subheader("⚡ AI Insights")
        # Run prediction
        pipeline = models[selected_model_name]
        
        with st.spinner("Analyzing transaction..."):
            time.sleep(0.8) # Fake delay for dramatic visual effect 
            prediction = pipeline.predict(transaction_data)[0]
            probability = pipeline.predict_proba(transaction_data)[0][1] * 100
            
        # Display Results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.markdown(f'<div class="fraud-alert"><h3>⛔ FRAUD ALERT!</h3><p>This transaction has signs of fraud and has been safely blocked by the system.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="normal-alert"><h3>✔️ SAFE TRANSACTION</h3><p>The system does not detect any risk, allowing the money to pass through.</p></div>', unsafe_allow_html=True)
                
        with res_col2:
            st.metric("Fraud Probability", f"{probability:.2f}%")
            if true_label == 1:
                st.warning(f"🔎 System label check: THIS IS A REAL FRAUD CASE")
            else:
                st.success(f"🔎 System label check: NORMAL USER")
                
            if prediction == true_label:
                st.write("🎉 **Perfect! A.I has blocked the gate ACCURATELY!**")
            else:
                st.write("❌ **Oops! A.I has processed this case WRONG.**")
