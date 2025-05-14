import os
import glob
import time
import io
import base64
import torch
import clip
from PIL import Image
from pinecone import Pinecone
from flask import Flask, render_template, request, url_for
from flask import send_from_directory
import torchvision.transforms as transforms

app = Flask(__name__)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_71n7ss_GtBhd6BQNMBVWRB6H71NXjbn8PJifw9KrUrtdX4phnndXeMpP6oFiYHQDJqmJN8")
index = pc.Index("fashion-search")

# Define the base directory where images are stored
IMAGE_BASE_PATH = r"D:\Akila\BD\Big data\preprocessed"


def encode_text(query_text):
    """Extracts embedding from a text query using CLIP."""
    text_tokens = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).cpu().numpy().flatten()
    return text_embedding


def encode_image(image_path):
    """Load image and transform it for embedding extraction."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 🔹 Load image correctly
    img = Image.open(image_path).convert("RGB")  # Open image as RGB
    img = transform(img)  # Apply transforms
    return img


def search_fashion(query_text=None, image=None, top_k=10):
    query_vector = None

    if query_text and image:
        text_embedding = encode_text(query_text)
        image_embedding = encode_image(image)
        query_vector = (text_embedding + image_embedding) / 2  
    elif query_text:
        query_vector = encode_text(query_text)
    elif image:
        query_vector = encode_image(image)
    else:
        return None

    results = index.query(vector=query_vector.tolist(), top_k=top_k, include_metadata=True)

    # Convert full paths to relative URLs
    for res in results["matches"]:
        if "image_path" in res["metadata"]:
            full_path = res["metadata"]["image_path"]
            relative_path = full_path.replace(IMAGE_BASE_PATH, "").lstrip("\\/")  # Make it relative
            res["metadata"]["image_url"] = f"/images/{relative_path}"  # Create a valid Flask URL

    
    return results



def find_image_in_subdirectories(image_filename):
    """Search for the image inside all subdirectories under 'preprocessed'."""
    filename_only = os.path.basename(image_filename)
    possible_paths = glob.glob(os.path.join(IMAGE_BASE_PATH, "**", filename_only), recursive=True)
    return possible_paths[0] if possible_paths else None


@app.route('/images/<path:filename>')
def serve_preprocessed_image(filename):
    return send_from_directory(IMAGE_BASE_PATH, filename)


UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static/images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def search():
    results = None
    search_image_path = None

    if request.method == "POST":
        query_text = request.form.get("query_text", "").strip()
        uploaded_image = request.files.get("uploaded_image")
        captured_image_data = request.form.get("captured_image")  # Base64 image data from camera

        image_data = None  # Image path to pass to `search_fashion`

        # 🔹 Handle Uploaded Image
        if uploaded_image and uploaded_image.filename:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_image.filename)
            uploaded_image.save(image_path)
            image_data = image_path
            search_image_path = url_for("static", filename=f"uploads/{uploaded_image.filename}")

        # 🔹 Handle Captured Image (Base64 to File)
        elif captured_image_data:
            import base64
            from datetime import datetime
            
            # Extract Base64 part & decode
            captured_image_data = captured_image_data.split(",")[1]
            image_bytes = base64.b64decode(captured_image_data)
            
            # Save the captured image
            filename = f"captured_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            image_data = image_path
            search_image_path = url_for("static", filename=f"uploads/{filename}")

        # 🔍 Call `search_fashion()` with text, image, or both
        results = search_fashion(
            query_text=query_text if query_text else None, 
            image=image_data if image_data else None, 
            top_k=10
        )

        # Ensure `results` has matches to avoid errors
        if results is None or "matches" not in results:
            results = {"matches": []}

        # 🔹 Update image paths for results
        for res in results["matches"]:
            image_relative_path = res["metadata"].get("image_path", "")
            res["metadata"]["image_path"] = url_for("static", filename=f"images/{image_relative_path}")

    return render_template("index.html", results=results, search_image_path=search_image_path)

if __name__ == "__main__":
    app.run(debug=True)