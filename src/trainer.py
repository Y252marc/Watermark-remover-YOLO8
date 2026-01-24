from ultralytics import YOLO
import os

def start_training():
    # Force the path to be inside your project folder
    project_root = os.getcwd()
    save_dir = os.path.join(project_root, 'runs', 'detect')

    print(f"[INFO] Saving models to: {save_dir}")

    model = YOLO('yolov8n.pt') 

    results = model.train(
        data='data.yaml',   # Path to our config file
        epochs=10,           # Start with 10 epochs to test. For real results, use 50-100.
        imgsz=640,          # Image size
        batch=16,           # Reduce this if you run out of GPU memory
        project=save_dir,   # <--- THIS IS THE FIX (Forces local save)
        name='watermark_v1' 
    )
    
    print(f"Training Complete! Saved to {save_dir}")

if __name__ == "__main__":
    start_training()
