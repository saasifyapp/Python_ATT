import os
import cv2
import numpy as np
import logging
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse
import insightface

# Suppress logs
logging.getLogger("insightface").setLevel(logging.ERROR)
cv2.setLogLevel(0)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Define local model path
model_path = "models/buffalo_l"  # Ensure this is the correct local path

# Enforce CPU usage (no GPU support)
ctx_id = -1  # Use CPU by default

# Initialize the face_model as None
face_model = None

def load_face_model():
    global face_model
    if face_model is None:
        # Check if the model files exist in the local directory
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            raise HTTPException(status_code=404, detail="Model files not found at the specified path.")
        
        try:
            # Load the face model from the local directory
            face_model = insightface.app.FaceAnalysis(name='buffalo_l')
            face_model.prepare(ctx_id=ctx_id)  # Initialize with CPU
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Decode base64 to image
def decode_base64_to_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_data = base64.b64decode(base64_string)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        return cv2.resize(img, (640, 640)) if img is not None else None
    except Exception:
        return None

# Extract embedding from image
def extract_embedding(base64_string):
    load_face_model()  # Lazy load model only when needed

    img = decode_base64_to_image(base64_string)
    if img is None:
        return None

    faces = face_model.get(img)
    if not faces:
        return None

    embedding = faces[0].embedding.tolist()
    embedding_shape = np.array(embedding).shape

    # Ensure the embedding has the correct dimension
    if embedding_shape == (512,):
        return embedding
    else:
        return None

# Cosine similarity function
def cosine_similarity_np(ref_embedding, img_embedding):
    dot_product = np.dot(ref_embedding, img_embedding)
    norm_product = np.linalg.norm(ref_embedding) * np.linalg.norm(img_embedding)
    similarity = dot_product / norm_product
    confidence = (similarity + 1) / 2 * 100  # Convert to percentage (0-100%)
    return confidence

# FastAPI app initialization
app = FastAPI()

class ImageData(BaseModel):
    images: List[str]  # List of base64 strings

class LiveImageData(BaseModel):
    image: str  # Single base64 image string

class StoredEmbedding(BaseModel):
    user_id: str
    name: str
    section: str
    standard_division: str
    embedding: List[List[float]]  # Ensuring embedding as a list of list of floats

stored_embeddings = []

@app.post("/extract-embedding")
async def extract_embeddings(data: ImageData):
    with ThreadPoolExecutor(max_workers=5) as executor:
        embeddings = list(filter(None, executor.map(extract_embedding, data.images)))

    return {"embeddings": embeddings[:5]}  # Return only 5 embeddings

@app.post("/store-retrieve-embeddings")
async def store_embeddings(data: List[StoredEmbedding]):
    try:
        # Clear existing records
        stored_embeddings.clear()

        for embedding_data in data:
            for emb in embedding_data.embedding:
                if np.array(emb).shape == (512,):
                    stored_embeddings.append({
                        "user_id": embedding_data.user_id,
                        "name": embedding_data.name,
                        "section": embedding_data.section,
                        "standard_division": embedding_data.standard_division,
                        "embedding": emb,
                    })

        return {"message": "Embeddings stored successfully", "count": len(stored_embeddings), "data": stored_embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embedd-live-face")
async def embedd_live_face(data: LiveImageData):
    embedding = extract_embedding(data.image)
    if embedding is None:
        return {"error": "No face detected or embedding failed"}

    threshold = 75  # Adjust threshold as needed

    matching_users = []
    match_found = False
    for stored in stored_embeddings:
        confidence = cosine_similarity_np(stored["embedding"], embedding)
        if confidence > threshold:
            matching_users.append({
                "user_id": stored["user_id"],
                "name": stored["name"],
                "section": stored["section"],
                "standard_division": stored["standard_division"],
                "confidence": confidence,
                "live_face_embedding": embedding
            })
            match_found = True
            break  # Exit after the first match

    if match_found:
        return {"status": "Match Found", "matches": matching_users}
    else:
        return {
            "status": "No Match Found",
            "error": "No match found",
            "confidence": "highest",
            "live_face_embedding": embedding
        }

@app.get("/model-status")
async def model_status():
    if face_model is not None:
        return {"status": "Model loaded", "model_path": model_path}
    else:
        return JSONResponse(status_code=404, content={"message": "Model not loaded or failed to load."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
