import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Load the TrOCR model and processor
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

processor, model = load_model()

st.title("Image Text Extractor")
st.write("Upload an image to extract text from it using Hugging Face's OCR model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate text
    with st.spinner('Extracting text...'):
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("Extracted Text:")
    st.write(generated_text)

else:
    st.write("Please upload an image to proceed.")