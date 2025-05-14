import os
import json
import base64
from pinecone import Pinecone

# Pinecone Configuration
INDEX_NAME = "fashion-search"

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_71n7ss_GtBhd6BQNMBVWRB6H71NXjbn8PJifw9KrUrtdX4phnndXeMpP6oFiYHQDJqmJN8")

# Connect to index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"❌ Index '{INDEX_NAME}' not found. Run `initialize_pinecone.py` first.")
    exit()

index = pc.Index(INDEX_NAME)

# Path to embeddings
embedding_output_path = r"C:\Users\aarth\OneDrive\Desktop\Big data\embedding"
image_base_path = r"C:\Users\aarth\OneDrive\Desktop\Big data\preprocessed"

# Process all JSON files in the directory
for file_name in os.listdir(embedding_output_path):
    if file_name.endswith("_embeddings.json"):
        json_path = os.path.join(embedding_output_path, file_name)

        # Load JSON file
        with open(json_path, "r") as f:
            image_embeddings = json.load(f)

        # Convert to Pinecone format
        vectors = []
        for img_name, emb in image_embeddings.items():
            img_path = os.path.join(image_base_path, img_name)

            # Store image path in metadata (or base64 if needed)
            metadata = {"image_path": img_path}  # Change this if using a URL or base64

            # Add to Pinecone vectors list
            vectors.append((img_name, emb, metadata))

        # Upload embeddings with metadata
        try:
            index.upsert(vectors=vectors)
            print(f"✅ Uploaded {len(vectors)} embeddings from {file_name} to Pinecone.")
        except Exception as e:
            print(f"❌ Error uploading {file_name}: {e}")

print("✅ All embeddings uploaded successfully!")
