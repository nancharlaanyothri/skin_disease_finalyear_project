import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------
# Custom Dark CSS
# ---------------------------
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
    }
    .main {
        background-color: #0f172a;
        color: white;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #22d3ee;
    }
    .subtitle {
        font-size: 18px;
        color: #cbd5e1;
    }
    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.keras")

model = load_model()

class_names = ['acne', 'eczema', 'melanoma', 'psoriasis', 'ringworm']

# ---------------------------
# Header Section
# ---------------------------
st.markdown('<div class="title">Skin Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered early skin condition diagnosis</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------
# Patient Info Section
# ---------------------------
st.markdown("## 🧑 Patient Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name")
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    location = st.text_input("Location")
    uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return predicted_class, confidence

# ---------------------------
# Show Result
# ---------------------------
if st.button("🔍 Diagnose"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, width=300)

        result, confidence = predict_image(image)

        st.markdown("## 🧪 Diagnosis Result")

        st.markdown(f"""
        <div class="card">
        <h3 style="color:#22c55e;">Diagnosed with: {result.upper()}</h3>
        <p>Confidence: {confidence}%</p>
        <p>Please consult a certified dermatologist for medical confirmation.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Please upload an image.")