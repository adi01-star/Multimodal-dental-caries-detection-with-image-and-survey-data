from ultralytics import YOLO
import os
# Load YOLOv8 model with custom weights
def load_model():
    # Get the root project directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the path to the trained model
    model_path = os.path.join(BASE_DIR, "data", "best.pt")

    # Load the YOLO model
    return YOLO(model_path)
