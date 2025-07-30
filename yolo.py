# yolov11_inference.py

from ultralytics import YOLO
from PIL import Image

model = YOLO("yolo11n.pt")  

def detect_objects(image):
    results = model(image)

    result = results[0]
    class_ids = result.boxes.cls.int().tolist()
    class_names = model.names
    labels = [class_names[i] for i in class_ids]

    # Annotated image with bounding boxes
    annotated_img_np = result.plot()  # Returns a numpy array (BGR)

    # Convert BGR → RGB and to PIL for Streamlit
    annotated_img_rgb = Image.fromarray(annotated_img_np[..., ::-1])  

    return list(labels), annotated_img_rgb