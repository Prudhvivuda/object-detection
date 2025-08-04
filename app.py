# app.py

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from yolo import detect_objects_all_models
from pillow_heif import register_heif_opener

register_heif_opener()

def load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image.thumbnail((1024, 1024))
        return image
    except Exception as e:
        st.error(f"❌ Failed to load image: {e}")
        st.stop()

st.set_page_config(page_title="YOLO Comparison", layout="wide")
st.title("📷 Object Detection and Model Comparison")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "heic", "heif"])

if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("---")

    with st.spinner("Running object detection with all models..."):
        detection_results = detect_objects_all_models(np.array(image))

    # Show 3 models per row
    model_items = list(detection_results.items())
    cols_per_row = 3

    for i in range(0, len(model_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (model_name, (labels, annotated_img)) in zip(cols, model_items[i:i+cols_per_row]):
            with col:
                st.markdown(f"### `{model_name}`")
                if labels:
                    st.markdown("**Objects:**")
                    st.markdown(", ".join(labels))
                else:
                    st.warning("No objects detected.")
                st.image(annotated_img, use_container_width=True)
