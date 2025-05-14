import os
from PIL import Image
import torch
import clip

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
dataset_path = r"C:\Users\aarth\OneDrive\Desktop\Big data\Train"
output_path = r"C:\Users\aarth\OneDrive\Desktop\Big data\preprocessed"

# Create output directories
if not os.path.exists(output_path):
    os.makedirs(output_path)

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    output_category_path = os.path.join(output_path, category)

    if not os.path.exists(output_category_path):
        os.makedirs(output_category_path)

    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            output_img_path = os.path.join(output_category_path, img_name)

            try:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image)
                
                # Save preprocessed image
                image.save(output_img_path)
                print(f"Processed and saved: {output_img_path}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("✅ Image preprocessing completed!")
