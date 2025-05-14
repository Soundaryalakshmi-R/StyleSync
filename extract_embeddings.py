import os
import torch
import clip
import json
from PIL import Image
import numpy as np

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
preprocessed_path = r"C:\Users\aarth\OneDrive\Desktop\Big data\preprocessed"
embedding_output_path = r"C:\Users\aarth\OneDrive\Desktop\Big data\embedding"

# Create directory for embeddings
if not os.path.exists(embedding_output_path):
    os.makedirs(embedding_output_path)

def extract_features(image_path):
    """Extracts feature vector from an image using CLIP."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
    
    return features.cpu().numpy().flatten()

# Process all images in the dataset
for category in os.listdir(preprocessed_path):
    category_path = os.path.join(preprocessed_path, category)
    if os.path.isdir(category_path):
        category_embeddings = {}

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                features = extract_features(img_path)
                category_embeddings[img_name] = features.tolist()  # Convert NumPy array to list
            except Exception as e:
                print(f"❌ Error extracting features from {img_path}: {e}")

        # Save category embeddings as a JSON file
        json_path = os.path.join(embedding_output_path, f"{category}_embeddings.json")
        with open(json_path, "w") as f:
            json.dump(category_embeddings, f)

        print(f"✅ Saved embeddings for {category} in JSON format")

print("✅ Feature extraction completed!")
