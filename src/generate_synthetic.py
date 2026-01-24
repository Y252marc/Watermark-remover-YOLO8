import cv2
import numpy as np
import os
import random
import glob
from tqdm import tqdm

def create_synthetic_data():
    # --- CONFIGURATION ---
    # 1. Where are your raw LOGOS?
    logo_dirs = [
        r'assets/LogoDataset/Logos', 
        r'assets/logos/independent',
        r'assets/logos/combined'
        r'/assets/my_screenshots/screenshots'
    ]
    
    # 2. Where are background images? (Using your existing Kaggle dataset)
    background_dir = r'assets/datasets/images/train'
    
    # 3. Where to save the NEW fake dataset?
    output_img_dir = r'assets/synthetic_dataset/images/train'
    output_lbl_dir = r'assets/synthetic_dataset/labels/train'
    
    # How many fake images do you want to create?
    NUM_IMAGES_TO_GENERATE = 10000 
    # ---------------------

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    # 1. Collect all Logos and Backgrounds
    print("[INFO] Loading file lists...")
    logo_paths = []
    for d in logo_dirs:
        logo_paths.extend(glob.glob(os.path.join(d, "*.*")))
    
    # Filter for images only
    logo_paths = [f for f in logo_paths if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    bg_paths = glob.glob(os.path.join(background_dir, "*.jpg")) + glob.glob(os.path.join(background_dir, "*.png"))

    if not logo_paths:
        print("[ERROR] No logos found! Check your paths in the script.")
        return
    if not bg_paths:
        print("[ERROR] No background images found! Check paths.")
        return

    print(f"[INFO] Found {len(logo_paths)} logos and {len(bg_paths)} backgrounds.")

    # 2. Generation Loop
    print(f"[INFO] Generating {NUM_IMAGES_TO_GENERATE} synthetic training images...")
    
    for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
        # A. Load random background
        bg_path = random.choice(bg_paths)
        bg = cv2.imread(bg_path)
        if bg is None: continue
        h_bg, w_bg = bg.shape[:2]

        # B. Load random logo
        logo_path = random.choice(logo_paths)
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED) # Load with Alpha if exists
        if logo is None: continue

        # Handle Logo Transparency
        if logo.shape[2] == 4:
            # Separate channels
            b, g, r, alpha = cv2.split(logo)
            logo_rgb = cv2.merge((b, g, r))
            mask = alpha
        else:
            logo_rgb = logo
            # Create a simple mask (assuming black/white background if not transparent)
            # Or just make it fully opaque
            mask = np.ones(logo.shape[:2], dtype=np.uint8) * 255

        # C. Resize Logo (Random size between 10% and 40% of background width)
        scale = random.uniform(0.1, 0.4)
        new_w = int(w_bg * scale)
        aspect_ratio = logo_rgb.shape[1] / logo_rgb.shape[0]
        new_h = int(new_w / aspect_ratio)
        
        try:
            logo_resized = cv2.resize(logo_rgb, (new_w, new_h))
            mask_resized = cv2.resize(mask, (new_w, new_h))
        except:
            continue # Skip if resizing fails

        # D. Random Position
        if h_bg - new_h <= 0 or w_bg - new_w <= 0: continue # Logo too big
        
        y_offset = random.randint(0, h_bg - new_h)
        x_offset = random.randint(0, w_bg - new_w)

        # E. Paste Logo using Mask
        roi = bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        
        # Blend: (Logo * alpha) + (Background * (1-alpha))
        mask_inv = cv2.bitwise_not(mask_resized)
        
        # Black-out the area of logo in ROI
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img_fg = cv2.bitwise_and(logo_resized, logo_resized, mask=mask_resized)
        # Put logo in ROI
        dst = cv2.add(img_bg, img_fg)
        
        bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = dst

        # F. Save Image
        filename = f"syn_{i:04d}.jpg"
        save_img_path = os.path.join(output_img_dir, filename)
        cv2.imwrite(save_img_path, bg)

        # G. Save Label (YOLO Format: class x_center y_center width height)
        # Normalize coordinates (0-1)
        x_center = (x_offset + new_w / 2) / w_bg
        y_center = (y_offset + new_h / 2) / h_bg
        width_norm = new_w / w_bg
        height_norm = new_h / h_bg

        save_lbl_path = os.path.join(output_lbl_dir, filename.replace('.jpg', '.txt'))
        with open(save_lbl_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {width_norm} {height_norm}\n")

    print("[SUCCESS] Synthetic dataset created at assets/synthetic_dataset")

if __name__ == "__main__":
    create_synthetic_data()