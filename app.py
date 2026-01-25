import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add 'src' to path so we can import the video processor
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.process_video import process_video

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def process_image(img_path, output_path, model_path):
    # Image Logic (Using v14 Model)
    print(f"[INFO] Using Image Model: {model_path}")
    model = YOLO(model_path)
    frame = cv2.imread(img_path)
    if frame is None: return

    # Standard settings for images
    results = model(frame, conf=0.25, verbose=False)[0]
    
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype="uint8")
    found = False
    
    for box in results.boxes:
        found = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Simple padding for images
        pad = 10
        cv2.rectangle(mask, (max(0,x1-pad), max(0,y1-pad)), (min(width,x2+pad), min(height,y2+pad)), 255, -1)

    if found:
        # Telea is often better for static images than Navier-Stokes
        cleaned = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(output_path, cleaned)
        print(f"[SUCCESS] Image saved: {output_path}")
    else:
        print("[INFO] No watermark found.")

def main():
    print("=== INTELLIGENT WATERMARK REMOVER (Hybrid Engine) ===")
    print("   [v14] Optimized for Images")
    print("   [v16] Optimized for Video (Aggressive Mode)")
    print("=====================================================\n")

    file_path = input("Drag and drop file here:\n").strip().replace('"', '')
    
    if not os.path.exists(file_path):
        print("[ERROR] File not found.")
        input("Press Enter...")
        return

    # Define Output
    folder = os.path.dirname(file_path)
    name, ext = os.path.splitext(os.path.basename(file_path))
    output_path = os.path.join(folder, f"cleaned_{name}{ext}")

    # Determine Type & Select Internal Model
    ext = ext.lower()
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    if ext in video_exts:
        print("\n>>> Mode: VIDEO (Loading v16_master)")
        # Get path to bundled v16.pt
        model_file = resource_path("v16.pt")
        process_video(file_path, output_path, model_file)

    elif ext in image_exts:
        print("\n>>> Mode: IMAGE (Loading v14_baseline)")
        # Get path to bundled v14.pt
        model_file = resource_path("v14.pt")
        process_image(file_path, output_path, model_file)
    
    else:
        print(f"[ERROR] Unsupported file type: {ext}")

    input("\nDone. Press Enter to exit...")

if __name__ == "__main__":
    main()