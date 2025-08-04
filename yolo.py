# yolo.py

from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Preload all YOLO models from the 'models' directory
MODELS_DIR = "models"
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
MODELS = {model_name: YOLO(os.path.join(MODELS_DIR, model_name)) for model_name in model_files}

def detect_objects_all_models(image_np):
    """
    Run object detection on all preloaded models and return results.
    image_np: numpy array (RGB)
    Returns: Dict of model name -> (labels, annotated PIL image)
    """
    results_dict = {}

    for model_name, model in MODELS.items():
        results = model(image_np)
        result = results[0]
        class_ids = result.boxes.cls.int().tolist()
        class_names = model.names
        labels = [class_names[i] for i in class_ids]

        annotated_img_np = result.plot()
        annotated_img_rgb = Image.fromarray(annotated_img_np[..., ::-1])  # BGR -> RGB

        results_dict[model_name] = (labels, annotated_img_rgb)

    return results_dict
