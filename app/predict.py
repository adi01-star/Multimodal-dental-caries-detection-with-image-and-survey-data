import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from app.model import load_model
except ImportError:
    from model import load_model

import torch

try:
    from torchvision.ops import nms
except ImportError:
    def nms(boxes, scores, iou_threshold=0.8):
        if len(boxes) == 0:
            return torch.empty((0,), dtype=torch.long)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= iou_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.tensor(keep, dtype=torch.long)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

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
    model_instance = get_model()
    results = model_instance(image_np)
    
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
