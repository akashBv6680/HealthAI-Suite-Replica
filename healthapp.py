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
from PIL import Image
import io

# ==================================================
# ENV + PAGE
# ==================================================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="HealthAI", layout="wide")
st.title("🧠 HealthAI - Multilingual Clinical AI System")
st.caption("Clinical ML * Medical RAG * Imaging Chat * Sentiment * Translator")

# Initialize Session State
if "uploaded_image_b64" not in st.session_state:
    st.session_state.uploaded_image_b64 = None

# ==================================================
# MODEL LOADING
# ==================================================
MODELS = {}
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    """Load ML models with error handling"""
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
    "Imaging Upload",
    "Medical RAG Chat",
    "Sentiment",
    "Translator",
    "About"
])

# ==================================================
# TAB 0 - CLINICAL ASSESSMENT
# ==================================================
with tabs[0]:
    st.subheader("Patient Clinical Assessment")
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
        risk = "HIGH" if sbp >= 140 or sugar >= 126 else "MEDIUM" if bmi >= 25 or sbp >= 130 else "LOW"
        st.subheader(f"Disease Risk: {risk}")
        if risk == "LOW": st.success("Low risk patient")
        elif risk == "MEDIUM": st.warning("Moderate risk - monitoring recommended")
        else: st.error("High risk - clinical attention required")

# ==================================================
# TAB 1 - IMAGING UPLOAD
# ==================================================
with tabs[1]:
    st.subheader("Medical Image Upload")
    st.write("Upload an image here, then discuss it with the AI in the 'Medical RAG Chat' tab.")
    
    uploaded_file = st.file_uploader("Upload Medical Image (X-ray, MRI, etc.)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Ready for Chat", width=400)
        
        # Convert to Base64 for RAG context
        img_byte_arr = io.BytesIO()
        image.convert('RGB').save(img_byte_arr, format='JPEG')
        st.session_state.uploaded_image_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        st.success("Image uploaded! Switch to 'Medical RAG Chat' to ask questions about it.")

# ==================================================
# TAB 2 - MEDICAL RAG CHAT (INTEGRATED)
# ==================================================
with tabs[2]:
    st.subheader("Medical Knowledge Chatbot")
    
    if st.session_state.uploaded_image_b64:
        st.info("🖼️ Image context is active. You can ask questions about your uploaded scan.")
    
    query = st.text_input("Ask a medical question (or ask about your uploaded image):")
    
    if query and API_KEY:
        with st.spinner("Consulting AI..."):
            try:
                # Use Llama 3.2 90B Vision (The current active vision-capable model on Groq)
                content = [{"type": "text", "text": f"User Medical Query: {query}"}]
                
                if st.session_state.uploaded_image_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.uploaded_image_b64}"}
                    })

                response = client.chat.completions.create(
                    model="llama-3.2-90b-vision-preview",
                    messages=[{"role": "user", "content": content}]
                )
                st.markdown("### Clinical Response")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Chat failed: {str(e)}")

# ==================================================
# TABS 3, 4, 5
# ==================================================
with tabs[3]:
    st.subheader("Sentiment Analysis")
    txt = st.text_area("Enter notes for sentiment analysis:")
    if txt and API_KEY:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": f"Analyze medical sentiment: {txt}"}])
        st.write(res.choices[0].message.content)

with tabs[4]:
    st.subheader("Medical Translator")
    src_txt = st.text_input("Text to translate:")
    t_lang = st.selectbox("Target Language:", ["Tamil", "Hindi", "Spanish", "French"])
    if src_txt and API_KEY:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": f"Translate to {t_lang}: {src_txt}"}])
        st.write(res.choices[0].message.content)

with tabs[5]:
    st.subheader("About")
    st.write("HealthAI v1.3 - Fixed Model Decommissioning Issue.")
