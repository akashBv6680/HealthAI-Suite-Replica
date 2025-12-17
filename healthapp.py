# ==================================================
# HEALTHAI - FINAL STABLE MULTILINGUAL CLINICAL AI SYSTEM
# ==================================================
import streamlit as st
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import json
import pickle
import base64
import tensorflow as tf
from tensorflow import keras

# ==================================================
# ENV + PAGE
# ==================================================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="HealthAI", layout="wide")
st.title("🧠 HealthAI - Multilingual Clinical AI System")
st.caption("Clinical ML * Imaging AI * Medical RAG * Sentiment * Translator")

# ==================================================
# MODEL LOADING (WITH GRACEFUL FALLBACK)
# ==================================================
MODELS = {}
MODEL_STATUS = {}
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    """Load ML models with error handling"""
    status = {}
    
    # List of expected model files
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
                    with open(model_path, 'r') as f:
                        MODELS[model_name] = json.load(f)
                    status[model_name] = "✓ Loaded"
                elif model_file.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        MODELS[model_name] = pickle.load(f)
                    status[model_name] = "✓ Loaded"
            else:
                status[model_name] = "✗ Not found"
        except Exception as e:
            status[model_name] = f"ERROR: {str(e)[:30]}"
    
    return status

# Load models on startup
MODEL_STATUS = load_models()

if not API_KEY:
    st.warning("GROQ_API_KEY not configured. Some features will be limited.")
else:
    from groq import Groq
    client = Groq(api_key=API_KEY)

# ==================================================
# UI TABS
# ==================================================
tabs = st.tabs([
    "Clinical Assessment",
    "Imaging Diagnosis",
    "Medical RAG",
    "Sentiment",
    "Translator",
    "About"
])

# ==================================================
# TAB 0 - CLINICAL ASSESSMENT
# ==================================================
with tabs[0]:
    st.subheader("Patient Clinical Assessment")
    st.write("Enter patient vitals to assess disease risk.")
    
    with st.form("clinical_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            hr = st.number_input("Heart Rate", 40, 150, 80)
        with c2:
            sbp = st.number_input("Systolic BP", 80, 250, 120)
            dbp = st.number_input("Diastolic BP", 40, 150, 80)
            sugar = st.number_input("Blood Sugar", 50, 400, 110)
            chol = st.number_input("Cholesterol", 100, 400, 180)
        
        analyze = st.form_submit_button("Analyze")
    
    if analyze:
        if sbp >= 140 or sugar >= 126:
            risk, color = "HIGH", "red"
        elif bmi >= 25 or sbp >= 130:
            risk, color = "MEDIUM", "orange"
        else:
            risk, color = "LOW", "green"
        
        st.markdown("---")
        st.subheader("Clinical Assessment Results")
        st.markdown(f"**Disease Risk: {risk}**")
        
        if risk == "LOW":
            st.success("Low risk patient")
        elif risk == "MEDIUM":
            st.warning("Moderate risk - monitoring recommended")
        else:
            st.error("High risk - clinical attention required")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Age Group", "Adult" if age >= 18 else "Minor")
        with c2:
            bmi_cat = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            st.metric("BMI Category", bmi_cat)
        with c3:
            st.metric("Risk Level", risk)

# ==================================================
# TAB 1 - IMAGING DIAGNOSIS
# ==================================================
with tabs[1]:
    st.subheader("Medical Image Analysis with Vision Language Model")
    st.write("Upload medical images for AI analysis using Groq VLM.")
    
    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", width=400)
        
        if not API_KEY:
            st.warning("Configure GROQ_API_KEY to enable VLM analysis.")
        else:
            if st.button("Analyze Image with Groq VLM"):
                with st.spinner("Analyzing image..."):
                    try:
                        image_data = image.convert('RGB')
                        import io
                        img_byte_arr = io.BytesIO()
                        image_data.save(img_byte_arr, format='JPEG')
                        img_byte_arr.seek(0)
                        base64_image = base64.b64encode(img_byte_arr.read()).decode('utf-8')
                        
                        message = client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct-90b",
                            max_tokens=1024,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": base64_image,
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": "Analyze this medical image. Provide: 1) Image type 2) Key findings 3) Observations 4) Next steps. Be clinical."
                                        }
                                    ],
                                }
                            ],
                        )
                        
                        analysis = message.choices[0].message.content
                        st.markdown("---")
                        st.subheader("VLM Analysis Results")
                        st.write(analysis)
                        st.success("Analysis completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)[:100]}")
                        st.info("Ensure GROQ_API_KEY supports vision models.")
    
    st.subheader("Medical Question Answering")
    if API_KEY:
        lang = st.selectbox("Choose language", ["English", "Tamil", "Hindi", "Spanish", "French"])
        question = st.text_input("Ask a medical question")
        
        if question:
            with st.spinner("Generating answer..."):
                try:
                    prompt = f"Answer in {lang}: {question}"
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.write(response.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("Configure GROQ_API_KEY to use this feature")

# ==================================================
# TAB 2 - MEDICAL RAG
# ==================================================
with tabs[2]:
    st.subheader("Medical Knowledge Base RAG")
    st.write("Retrieve medical information from knowledge base.")
    
    if API_KEY:
        query = st.text_input("Enter medical query")
        
        if query:
            with st.spinner("Retrieving information..."):
                try:
                    prompt = f"""Based on medical knowledge, answer this clinical query:
{query}

Provide evidence-based response with references if possible."""
                    
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.success(response.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("Configure GROQ_API_KEY to use this feature")

# ==================================================
# TAB 3 - SENTIMENT ANALYSIS
# ==================================================
with tabs[3]:
    st.subheader("Text Sentiment Analysis")
    
    if API_KEY:
        text_input = st.text_area("Enter text", height=100)
        
        if text_input:
            with st.spinner("Analyzing..."):
                try:
                    prompt = f"Classify sentiment (Positive/Neutral/Negative): {text_input}"
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.success(response.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("Configure GROQ_API_KEY to use this feature")

# ==================================================
# TAB 4 - TRANSLATOR
# ==================================================
with tabs[4]:
    st.subheader("Multilingual Translator")
    
    if API_KEY:
        text = st.text_input("Text to translate")
        target_lang = st.selectbox("Translate to", ["Tamil", "Hindi", "Spanish", "French"])
        
        if text:
            with st.spinner("Translating..."):
                try:
                    prompt = f"Translate to {target_lang}: {text}"
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.write(response.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("Configure GROQ_API_KEY to use this feature")

# ==================================================
# TAB 5 - ABOUT
# ==================================================
with tabs[5]:
    st.subheader("About HealthAI Suite")
    st.write("""
### Mission
Provide accessible, multilingual clinical AI for healthcare professionals.

### Features
- Clinical Assessment based on vital signs
- Medical Image Analysis (VLM)
- Medical RAG for knowledge retrieval
- Medical Q&A in multiple languages
- Sentiment analysis
- Multilingual translation

### Disclaimer
This is decision-support only. Always consult healthcare professionals.

### Setup
1. Run: `python models/download_models.py`
2. Add GROQ_API_KEY to Streamlit Secrets
3. Redeploy

Version: 1.1 (Beta)
    """)
    st.divider()
    st.caption("HealthAI Suite | Powered by Streamlit & Groq")
