import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title=" Skin Disease Prediction",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------------------------
# LOAD HUGGINGFACE MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("Ateeqq/skin-disease-prediction-exp-v1")
    model = AutoModelForImageClassification.from_pretrained("Ateeqq/skin-disease-prediction-exp-v1")
    return processor, model

processor, model = load_model()

# -------------------------------------------------
# THEME TOGGLE
# -------------------------------------------------
mode = st.sidebar.radio("Select Theme", ["Dark 🌙", "Light ☀️"])

if mode == "Dark 🌙":
    bg = "#0f172a"
    text = "white"
else:
    bg = "#f8fafc"
    text = "black"

st.markdown(f"""
<style>
.main {{
    background-color: {bg};
    color: {text};
}}
.card {{
    background-color: #1e293b;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🧬 AI Skin Disease Detection")
st.caption("Powered by HuggingFace Deep Learning Model")

# -------------------------------------------------
# SIDEBAR PATIENT INFO
# -------------------------------------------------
st.sidebar.header("Patient Info")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def predict(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    label = model.config.id2label[predicted_class.item()]
    conf = round(confidence.item() * 100, 2)

    return label, conf, probs[0].numpy()

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
if st.button("🔍 Analyze"):

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=300)

        result, confidence, probabilities = predict(image)

        st.markdown("## 🧪 Diagnosis Result")

        st.markdown(f"""
        <div class="card">
            <h2 style="color:#22c55e;">{result.upper()}</h2>
            <h4>Confidence: {confidence}%</h4>
            <p>Please consult a dermatologist for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability Chart
        labels = list(model.config.id2label.values())

        prob_data = {
            labels[i]: float(probabilities[i]) * 100
            for i in range(len(labels))
        }

        st.subheader("Prediction Confidence for All Classes")
        st.bar_chart(prob_data)

    else:
        st.warning("Please upload an image first.")