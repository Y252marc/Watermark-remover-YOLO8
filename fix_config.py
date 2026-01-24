import os
import yaml

# 1. Get the absolute path to your project root
# This defines E:\User\ASUS\Documents\programmer boy\yolo-work\watermark detector\WatermarkRemover
base_dir = os.getcwd()

# 2. Define the absolute paths to your data
# Make sure these folders actually exist!
train_path = os.path.join(base_dir, 'assets', 'dataset', 'images', 'train')
val_path = os.path.join(base_dir, 'assets', 'dataset', 'images', 'val')

# 3. Create the dictionary structure
data_config = {
    'path': os.path.join(base_dir, 'assets', 'dataset'), # Root dir for dataset
    'train': train_path,
    'val': val_path,
    'names': {
        0: 'watermark'
    }
}

# 4. Write to data.yaml
output_file = 'data.yaml'
with open(output_file, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ Generated {output_file} successfully!")
print(f"Train Path: {train_path}")
print(f"Val Path:   {val_path}")

# 5. Quick check if folders exist
if not os.path.exists(train_path):
    print("\n❌ CRITICAL WARNING: The train folder was not found!")
    print(f"   I looked here: {train_path}")
    print("   Please check if your folder structure matches 'assets/dataset/images/train'")
else:
    print("\n✅ Train folder found. You are ready to train.")