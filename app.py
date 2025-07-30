import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from yolo import detect_objects
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

st.set_page_config(page_title="YOLOv11 Detector", layout="centered")
st.title("📷 YOLOv11 Object Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "heic", "heif"])


if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        detected, annotated_image = detect_objects(np.array(image))

    if detected:
        st.success("Detected objects:")
        for obj in detected:
            st.write(f"✅ {obj}")
        st.image(annotated_image, caption="Detected Image", use_container_width=True)

    else:
        st.warning("No objects detected.")

