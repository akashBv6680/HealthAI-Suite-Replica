# ==================================================
# HEALTHAI - UPDATED FOR GEMINI AI (DEC 2025)
# ==================================================
import streamlit as st
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import json
import pickle
import base64
from PIL import Image
import google.generativeai as genai

# ==================================================
# ENV + PAGE CONFIG
# ==================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="HealthAI", layout="wide", page_icon="🧠")
st.title("🧠 HealthAI - Multilingual Clinical AI System")
st.caption("Clinical ML * Imaging AI * Medical RAG * Sentiment * Translator")

# Initialize Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.warning("GEMINI_API_KEY not found. Please check your .env file.")

# ==================================================
# MODEL LOADING (Legacy ML Models)
# ==================================================
MODELS = {}
MODELS_DIR = "models"

@st.cache_resource
def load_ml_models():
    status = {}
    expected_models = {
        "xgboost_disease_model.json": "XGBoost Disease Model",
        "association_rules.json": "Association Rules",
        "los_scaler.pkl": "LOS Scaler",
        "kmeans_cluster_model.pkl": "KMeans Clustering",
        "cluster_scaler_final.pkl": "Cluster Scaler"
    }
    for model_file, model_name in expected_models.items():
        model_path = os.path.join(MODELS_DIR, model_file)
        try:
            if os.path.exists(model_path):
                if model_file.endswith('.json'):
                    with open(model_path, 'r') as f: MODELS[model_name] = json.load(f)
                elif model_file.endswith('.pkl'):
                    with open(model_path, 'rb') as f: MODELS[model_name] = pickle.load(f)
                status[model_name] = "✓ Loaded"
            else:
                status[model_name] = "✗ Not found"
        except Exception as e:
            status[model_name] = f"ERROR: {str(e)[:20]}"
    return status

MODEL_STATUS = load_ml_models()

# ==================================================
# UI TABS
# ==================================================
tabs = st.tabs(["Clinical Assessment", "Imaging Diagnosis", "Medical RAG", "Sentiment", "Translator", "About"])

# ==================================================
# TAB 0 - CLINICAL ASSESSMENT (Keep logic as-is)
# ==================================================
with tabs[0]:
    st.subheader("Patient Clinical Assessment")
    with st.form("clinical_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        with c2:
            sbp = st.number_input("Systolic BP", 80, 250, 120)
            sugar = st.number_input("Blood Sugar", 50, 400, 110)
        analyze = st.form_submit_button("Analyze")
    
    if analyze:
        risk = "HIGH" if (sbp >= 140 or sugar >= 126) else "MEDIUM" if (bmi >= 25 or sbp >= 130) else "LOW"
        st.metric("Disease Risk", risk)

# ==================================================
# TAB 1 - IMAGING DIAGNOSIS (GEMINI VLM)
# ==================================================
with tabs[1]:
    st.subheader("Medical Image Analysis (Gemini VLM)")
    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and API_KEY:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=400)
        
        if st.button("Analyze Image with Gemini 2.5 Flash"):
            with st.spinner("Processing medical vision data..."):
                try:
                    # Gemini handles multimodal inputs naturally
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = "Analyze this medical image clinically. Provide: 1) Image type 2) Key findings 3) Observations 4) Recommended next steps."
                    
                    response = model.generate_content([prompt, img])
                    st.markdown("### Analysis Results")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Vision Error: {e}")

# ==================================================
# TAB 2 - MEDICAL RAG (GEMINI 2.5 PRO)
# ==================================================
with tabs[2]:
    st.subheader("Medical Knowledge Base (Gemini 2.5 Pro)")
    query = st.text_input("Enter clinical query for RAG simulation")
    if query and API_KEY:
        with st.spinner("Searching medical literature..."):
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(f"Answer this medical query using evidence-based data: {query}")
            st.info(response.text)

# ==================================================
# TAB 3 - SENTIMENT
# ==================================================
with tabs[3]:
    st.subheader("Patient Sentiment Analysis")
    text_input = st.text_area("Patient feedback/notes")
    if text_input and API_KEY:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(f"Classify sentiment (Positive/Neutral/Negative): {text_input}")
        st.success(f"Result: {response.text.strip()}")

# ==================================================
# TAB 4 - TRANSLATOR
# ==================================================
with tabs[4]:
    st.subheader("Multilingual Clinical Translator")
    text = st.text_input("Text to translate")
    target_lang = st.selectbox("Language", ["Tamil", "Hindi", "Spanish", "French"])
    if text and API_KEY:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(f"Translate this clinical text to {target_lang}: {text}")
        st.code(response.text)

# ==================================================
# TAB 5 - ABOUT
# ==================================================
with tabs[5]:
    st.write("### HealthAI Suite v1.5 (Gemini Edition)")
    st.write("Now powered by Gemini 2.5 for superior medical reasoning and multimodal vision.")
    st.json(MODEL_STATUS)
