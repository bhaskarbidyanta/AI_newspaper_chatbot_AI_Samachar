import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
import re

st.title("📜 Newspaper OCR - Extract & Format Articles")

uploaded_file = st.file_uploader("Upload a newspaper article image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # Preprocess the image
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    st.image(thresh, caption="Preprocessed Image", use_column_width=True)
    
    # Extract text
    config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(thresh, config=config)

    # Clean and format text
    def clean_text(text):
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraphs
        text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
        return text.strip()
    
    formatted_text = clean_text(extracted_text)

    st.subheader("📰 Extracted & Formatted Text")
    st.text_area("", formatted_text, height=400)

