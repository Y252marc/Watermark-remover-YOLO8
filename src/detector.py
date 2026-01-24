import cv2
from ultralytics import YOLO

class WatermarkDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        
        Initialize the YOLO detector.
        If no custom model is provided, it downloads the standard 'nano' model (yolov8n.pt).
        """
        print(f"[INFO] Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, frame, confidence_threshold=0.5):
        """
        Scans a single video frame for objects.
        Returns a list of bounding boxes: [(x1, y1, x2, y2), ...]
        """
        # Run inference on the frame
        # verbose=False keeps the terminal clean
        results = self.model(frame, verbose=False)[0] 

        boxes = []
        
        # Iterate through detections
        for box in results.boxes:
            # Check confidence (how sure the AI is)
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            
            # Get coordinates (x1, y1, x2, y2)
            # x1, y1 = top-left corner
            # x2, y2 = bottom-right corner
            coords = box.xyxy[0].tolist() 
            x1, y1, x2, y2 = map(int, coords)
            
            boxes.append((x1, y1, x2, y2))
            
        return boxes

# Quick test if you run this file directly
if __name__ == "__main__":
    # Create a dummy black image to test the code logic
    import numpy as np
    dummy_frame = np.zeros((640, 640, 3), dtype="uint8")
    
    detector = WatermarkDetector()
    detections = detector.detect(dummy_frame)
    print(f"Test run complete. Detections found: {len(detections)}")