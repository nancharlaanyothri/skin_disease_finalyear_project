import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

st.set_page_config(page_title="AI Skin Disease Detection")

st.title("🧬 AI Skin Disease Detection")
st.write("Powered by HuggingFace Model")

@st.cache_resource
def load_classifier():
    return pipeline(
        task="image-classification",
        model="Ateeqq/skin-disease-prediction-exp-v1",
        framework="pt"
    )

classifier = load_classifier()

uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        result = classifier(image)

    st.write("### Prediction:")
    st.write(result[0]["label"])
    st.write("Confidence:", round(result[0]["score"] * 100, 2), "%")