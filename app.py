import streamlit as st
from PIL import Image
from transformers import pipeline
import numpy as np

st.set_page_config(page_title="AI Skin Disease Detection", layout="wide")

st.title("🧬 AI Skin Disease Detection")
st.caption("Powered by HuggingFace Model")

# ------------------------------------------------
# LOAD MODEL USING PIPELINE (LIGHTER)
# ------------------------------------------------
@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="Ateeqq/skin-disease-prediction-exp-v1"
    )

classifier = load_classifier()

uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

if st.button("Analyze"):

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=300)

        results = classifier(image)

        top_result = results[0]
        label = top_result["label"]
        confidence = round(top_result["score"] * 100, 2)

        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence}%")

        st.subheader("All Predictions")

        for r in results:
            st.write(f"{r['label']} : {round(r['score']*100,2)}%")

    else:
        st.warning("Please upload an image.")