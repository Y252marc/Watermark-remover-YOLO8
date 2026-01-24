import cv2
import os
import subprocess
from ultralytics import YOLO
import numpy as np

def process_video(video_path, output_path, model_path):
    # --- CONFIGURATION ---
    CONFIDENCE_THRESHOLD = 0.25  # Lowered: Catches blurry/faint watermarks
    MASK_PADDING = 15            # Increased: Wipes a larger area around the logo
    # ---------------------

    # 1. Validation
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    # Temporary output for the video-only file (no audio yet)
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")

    # 2. Initialize Model
    print(f"[INFO] Loading Model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Processing {total_frames} frames (Aggressive Mode)...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # --- A. AGGRESSIVE DETECTION ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        # --- B. MASKING ---
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        detections_found = False
        
        for box in results.boxes:
            detections_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Aggressive Padding
            x1 = max(0, x1 - MASK_PADDING)
            y1 = max(0, y1 - MASK_PADDING)
            x2 = min(width, x2 + MASK_PADDING)
            y2 = min(height, y2 + MASK_PADDING)
            
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
        # --- C. REMOVE ---
        if detections_found:
            # Navier-Stokes is smoother for video
            cleaned_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
        else:
            cleaned_frame = frame 

        out.write(cleaned_frame)
        
        if frame_count % 50 == 0:
            print(f"Processed: {frame_count}/{total_frames}")

    cap.release()
    out.release()
    
    # --- D. AUDIO MERGE (FFMPEG) ---
    print("[INFO] Merging Audio...")
    
    # Command to copy audio from 'video_path' to 'temp_video_path' -> save as 'output_path'
    command = [
        "ffmpeg", "-y",                 # Overwrite output
        "-i", temp_video_path,          # Input 1: Clean Video
        "-i", video_path,               # Input 2: Original Audio
        "-c:v", "copy",                 # Copy video stream (don't re-encode)
        "-c:a", "aac",                  # Encode audio to AAC
        "-map", "0:v:0",                # Take video from Input 1
        "-map", "1:a:0",                # Take audio from Input 2
        "-shortest",                    # Stop when shortest input ends
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[SUCCESS] Final video with Audio: {output_path}")
        # Clean up temp file
        os.remove(temp_video_path)
    except FileNotFoundError:
        print("[ERROR] FFmpeg not found! Install it with: sudo apt install ffmpeg")
    except Exception as e:
        print(f"[ERROR] Audio merge failed: {e}")

if __name__ == "__main__":
    # CHECK YOUR PATHS
    input_video = "assets/input_videos/test_video.mp4"
    output_video = "assets/output_videos/cleaned_result.mp4"
    my_model = "runs/detect/v16_master/weights/best.pt"
    
    process_video(input_video, output_video, my_model)