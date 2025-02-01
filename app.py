import streamlit as st
import pytesseract
from PIL import Image

st.title("Image to Text Extraction using OCR")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Extracting text...")
    extracted_text = pytesseract.image_to_string(image)

    st.text_area("Extracted Text:", extracted_text, height=300)
