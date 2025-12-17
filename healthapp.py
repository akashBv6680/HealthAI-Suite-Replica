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

# Initialize Session State for RAG context
if "vlm_analysis" not in st.session_state:
    st.session_state.vlm_analysis = ""

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
        st.write(f"**Disease Risk: {risk}**")
        
        if risk == "LOW": st.success("Low risk patient")
        elif risk == "MEDIUM": st.warning("Moderate risk - monitoring recommended")
        else: st.error("High risk - clinical attention required")

# ==================================================
# TAB 1 - IMAGING DIAGNOSIS (AUTO-TRIGGER)
# ==================================================
with tabs[1]:
    st.subheader("Medical Image Analysis (Auto-Analysis)")
    st.write("Upload an image to get an immediate AI assessment.")
    
    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"], key="vlm_uploader")
    
    if uploaded_file is not None:
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)
        
        if API_KEY:
            with st.spinner("AI is analyzing the image..."):
                try:
                    import io
                    img_byte_arr = io.BytesIO()
                    image.convert('RGB').save(img_byte_arr, format='JPEG')
                    base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    
                    response = client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Provide a detailed clinical analysis: Type, Findings, and Next Steps."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }]
                    )
                    
                    st.session_state.vlm_analysis = response.choices[0].message.content
                    st.success("Analysis Complete!")
                    st.markdown("---")
                    st.markdown(st.session_state.vlm_analysis)
                    st.info("💡 You can now ask specific questions about this analysis in the 'Medical RAG' tab.")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# ==================================================
# TAB 2 - MEDICAL RAG (WITH CONTEXT)
# ==================================================
with tabs[2]:
    st.subheader("Medical Knowledge Base & Follow-up")
    
    # Show context if image was analyzed
    if st.session_state.vlm_analysis:
        with st.expander("Reference Image Analysis Context"):
            st.write(st.session_state.vlm_analysis)
    
    query = st.text_input("Ask a medical question or follow up on your image analysis:")
    
    if query and API_KEY:
        with st.spinner("Searching knowledge base..."):
            try:
                # Build context-aware prompt
                context = f"Previous Image Analysis: {st.session_state.vlm_analysis}" if st.session_state.vlm_analysis else "No previous image context."
                
                full_prompt = f"""Context: {context}
                
                User Query: {query}
                
                Instruction: Answer the query based on medical evidence. If the query is related to the previous image analysis context provided above, integrate that information into your response."""
                
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": full_prompt}]
                )
                st.markdown("### Response")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")

# ==================================================
# TABS 3, 4, 5 (SENTIMENT, TRANSLATOR, ABOUT)
# ==================================================
with tabs[3]:
    st.subheader("Sentiment Analysis")
    txt = st.text_area("Patient feedback/notes:")
    if txt and API_KEY:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": f"Sentiment of: {txt}"}])
        st.success(res.choices[0].message.content)

with tabs[4]:
    st.subheader("Translator")
    src_txt = st.text_input("Translate this:")
    t_lang = st.selectbox("To:", ["Tamil", "Hindi", "Spanish", "French"])
    if src_txt and API_KEY:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": f"Translate to {t_lang}: {src_txt}"}])
        st.write(res.choices[0].message.content)

with tabs[5]:
    st.subheader("About")
    st.write("HealthAI v1.2 - Integrated Imaging & RAG System.")
