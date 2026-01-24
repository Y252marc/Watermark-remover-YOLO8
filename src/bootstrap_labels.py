from ultralytics import YOLO
import os
import glob

def auto_label_screenshots():
    # 1. Model Path
    model_path = 'runs/detect/v15_teacher2/weights/best.pt'
    
    # 2. UPDATED PATHS (The Fix)
    # We point to the 'screenshots' subfolder now
    images_dir = 'assets/my_screenshots/screenshots'
    labels_dir = 'assets/my_screenshots/screenshots' # Save labels next to images
    
    # Safety Check for the Model
    if not os.path.exists(model_path):
        print(f"[ERROR] Cannot find model at: {model_path}")
        return

    # Safety Check for the Images Folder
    if not os.path.exists(images_dir):
        print(f"[ERROR] Cannot find image folder at: {images_dir}")
        print("Check if you named the folder 'screenshot' (singular) or 'screenshots' (plural).")
        return

    print(f"[INFO] Loading Teacher Model: {model_path}")
    model = YOLO(model_path)
    
    # Find images
    image_files = glob.glob(os.path.join(images_dir, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"[INFO] Found {len(image_files)} screenshots. Starting auto-labeling...")
    
    count = 0
    for img_path in image_files:
        # conf=0.15 is low enough to catch faint watermarks
        results = model(img_path, conf=0.01, verbose=False)[0]
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(labels_dir, base_name + ".txt")
        
        # Write the label file
        with open(txt_path, 'w') as f:
            for box in results.boxes:
                cls = 0 
                x, y, w, h = box.xywhn[0]
                f.write(f"{cls} {x} {y} {w} {h}\n")
                
        if len(results.boxes) > 0:
            count += 1
            print(f"   [+] Labeled: {base_name}")
            
    print(f"\n[SUCCESS] Auto-labeled {count} images.")
    print(f"Check this folder for .txt files: {labels_dir}")

if __name__ == "__main__":
    auto_label_screenshots()