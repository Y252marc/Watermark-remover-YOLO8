import cv2
import os
import glob
from ultralytics import YOLO

def test_custom_model():
    # 1. PATH CONFIGURATION (CRITICAL)
    # This must match where your training just finished
    model_path = 'runs/detect/v16_master/weights/best.pt'
    
    # Path to your personal screenshots
    test_folder = 'assets/my_screenshots'
    
    # Where to save the results so we can look at them
    output_folder = 'assets/verification_results'
    os.makedirs(output_folder, exist_ok=True)

    # 2. Load your new AI
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find model at {model_path}")
        print("Please check the folder number in 'runs/detect/...'")
        return

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # 3. Find all images (jpg, png, jpeg)
    image_files = glob.glob(os.path.join(test_folder, "*.*"))
    # Filter only images
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"[ERROR] No images found in {test_folder}. Please add some screenshots!")
        return

    print(f"[INFO] Testing on {len(image_files)} images...")

    # 4. Run Inference
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Read image
        frame = cv2.imread(img_path)
        if frame is None: continue

        # Detect
        results = model(frame, verbose=False)[0]
        
        # Draw boxes
        for box in results.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw Red Box with Confidence Score
            label = f"Watermark: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save to output folder
        save_path = os.path.join(output_folder, "checked_" + filename)
        cv2.imwrite(save_path, frame)
        print(f"   Saved: {save_path}")

    print(f"\n[DONE] Go check the folder: {output_folder}")

if __name__ == "__main__":
    test_custom_model()