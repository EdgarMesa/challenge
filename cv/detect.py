import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import supervision as sv
from ultralytics import YOLO

app = FastAPI()

model = YOLO('yolo11s.pt')

# Map of classes based in COCO dataset
CLASSES_OF_INTEREST = {0: 'person', 2: 'car'}

@app.post('/detect')
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read image from the requests
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = np.array(image)
        
        results = model.predict(image, conf=0.4, iou=0.85, imgsz=640,
                                classes=list(CLASSES_OF_INTEREST.keys()))
        detections = sv.Detections.from_ultralytics(results[0])
        
        all_parsed_detections = []
        for i in range(len(detections)):
            detection = detections[i]
            # Extract the coordinates
            x1, y1, x2, y2 = detection.xyxy[0]
            confidence = float(detection.confidence[0])
            class_name = str(detection.data['class_name'][0])
            
            # Create a dictionary for the current detection with rounded values
            detection_dict = {
                'id': i,
                'x1': round(x1.item(), 4),
                'y1': round(y1.item(), 4),
                'x2': round(x2.item(), 4),
                'y2': round(y2.item(), 4),
                'confidence': round(confidence, 2),
                'class_name': class_name
            }
            all_parsed_detections.append(detection_dict)
        
        
        return JSONResponse(content={'detections': all_parsed_detections})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
