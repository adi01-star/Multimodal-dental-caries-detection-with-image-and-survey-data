import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import load_model
import torch
from torchvision.ops import nms

model = load_model()

def predict(image: Image.Image):
    """
    Runs the YOLO caries-detection pipeline on a single dental photo.

    Returns
    -------
    img_with_boxes : PIL.Image
        Copy of the input image with bounding boxes drawn on any detected
        caries.
    status : str
        "Caries Detected" or "No Caries Detected".
    confidence : float
        Highest detection confidence in [0, 1] if caries were detected,
        else 0.0. Used as the image pipeline's probability when combining
        with the lifestyle pipeline (see app/fusion.py).
    """
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Perform inference with YOLOv8 model
    results = model(image_np)
    
    # Convert the image back to a PIL image to draw bounding boxes
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    caries_detected = False  # Flag to check if caries are detected

    # Extract bounding boxes and confidence scores
    boxes = []
    scores = []
    
    for r in results:
        for bbox in r.boxes:
            cls_id = int(bbox.cls)  # Only "caries" class exists (class 0)
            
            if cls_id == 0:  # Draw bounding box only for "caries" class
                caries_detected = True
                x1, y1, x2, y2 = bbox.xyxy.tolist()[0]
                confidence = bbox.conf.tolist()[0]
                
                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)

    # Convert to tensors and apply NMS
    if boxes:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        iou_threshold = 0.8
        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
    
        # Load default font (or use a custom font if available)
        try:
            font = ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()  # Fallback

        # Draw bounding boxes and text
        for bbox, score in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"Caries ({score:.2f})", fill="red", font=font)

    # Check if caries were detected and set the corresponding output message
    if caries_detected:
        status = "Caries Detected"
        confidence = float(max(scores.tolist()))  # Return highest confidence value
    else:
        status = "No Caries Detected"
        confidence = 0.0
    
    return img_with_boxes, status, confidence
