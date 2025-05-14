import numpy as np
from PIL import Image
import torch
import clip
import pinecone

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Connect to Pinecone
pinecone.init(api_key="pcsk_71n7ss_GtBhd6BQNMBVWRB6H71NXjbn8PJifw9KrUrtdX4phnndXeMpP6oFiYHQDJqmJN8")
index = pinecone.Index("fashion-search")

def encode_text(query_text):
    """Extracts embedding from a text query using CLIP."""
    text_tokens = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).cpu().numpy().flatten()
    return text_embedding

def encode_image(image_path):
    """Extracts embedding from an image using CLIP."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor).cpu().numpy().flatten()
    return image_embedding

def search_fashion(query_text=None, image_path=None, top_k=5):
    """Search Pinecone using text, image, or both."""
    query_vector = None

    if query_text and image_path:
        text_embedding = encode_text(query_text)
        image_embedding = encode_image(image_path)
        query_vector = (text_embedding + image_embedding) / 2  # Combine text and image embeddings
    elif query_text:
        query_vector = encode_text(query_text)
    elif image_path:
        query_vector = encode_image(image_path)
    else:
        raise ValueError("Provide either text, image, or both.")

    # Query Pinecone
    results = index.query(vector=query_vector.tolist(), top_k=top_k, include_metadata=True)
    return results

# Example usage
if __name__ == "__main__":
    results = search_fashion(query_text="red dress", image_path="test_image.jpg", top_k=5)
    print("🔎 Search Results:", results)
