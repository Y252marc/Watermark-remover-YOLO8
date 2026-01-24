import cv2
import os
from src.detector import WatermarkDetector
from src.remover import WatermarkRemover

def main():
    # 1. Setup
    input_image_path = 'assets/test_image.jpg'
    
    if not os.path.exists(input_image_path):
        print(f"[ERROR] No image found at {input_image_path}")
        return

    # 2. Initialize Models
    print("[INIT] Loading Detector...")
    detector = WatermarkDetector(model_path='yolov8n.pt') 
    
    print("[INIT] Loading Remover...")
    remover = WatermarkRemover(method='telea')

    # 3. Load Image
    frame = cv2.imread(input_image_path)
    
    # 4. PIPELINE EXECUTION
    # Step A: Detect
    print("[1/2] Detecting objects...")
    boxes = detector.detect(frame)
    print(f"      Found {len(boxes)} objects.")

    # Step B: Remove
    print("[2/2] Removing objects...")
    cleaned_frame = remover.remove(frame, boxes)

    # 5. Visualization (Compare Side-by-Side)
    # Resize for better viewing if the image is huge
    scale = 0.5
    h, w = frame.shape[:2]
    new_dim = (int(w * scale), int(h * scale))
    
    small_original = cv2.resize(frame, new_dim)
    small_cleaned = cv2.resize(cleaned_frame, new_dim)
    
    # Stack images horizontally
    comparison = cv2.hconcat([small_original, small_cleaned])

    cv2.imshow("Original vs Processed", comparison)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    cv2.imwrite("assets/result_preview.jpg", cleaned_frame)
    print("[DONE] Result saved to assets/result_preview.jpg")

if __name__ == "__main__":
    main()