import cv2
import numpy as np
import insightface
import torch
import logging
import os
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Suppress logs
logging.getLogger("insightface").setLevel(logging.ERROR)
cv2.setLogLevel(0)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

ctx_id = -1
embedding_dimension = 512

# Global variable for lazy loading
face_model = None

def get_face_model():
    global face_model
    if face_model is None:
        model_dir = os.path.join("models", "buffalo_l")  # Adjust path as per Render's deployment
        face_model = insightface.app.FaceAnalysis(name='buffalo_l', root=model_dir)
        face_model.prepare(ctx_id=ctx_id)
    return face_model

app = FastAPI()

class ImageData(BaseModel):
    images: List[str]

class LiveImageData(BaseModel):
    image: str

class StoredEmbedding(BaseModel):
    user_id: str
    name: str
    section: str
    standard_division: str
    embedding: List[List[float]]

stored_embeddings = []

def decode_base64_to_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(img, (640, 640)) if img is not None else None
    except Exception:
        return None

def extract_embedding(base64_string):
    model = get_face_model()
    img = decode_base64_to_image(base64_string)
    if img is None:
        return None
    faces = model.get(img)
    if not faces:
        return None
    embedding = faces[0].embedding.tolist()
    if np.array(embedding).shape == (embedding_dimension,):
        return embedding
    return None

def cosine_similarity_np(ref_embedding, img_embedding):
    dot_product = np.dot(ref_embedding, img_embedding)
    norm_product = np.linalg.norm(ref_embedding) * np.linalg.norm(img_embedding)
    similarity = dot_product / norm_product
    confidence = (similarity + 1) / 2 * 100
    return confidence

@app.post("/extract-embedding")
async def extract_embeddings(data: ImageData):
    with ThreadPoolExecutor() as executor:
        embeddings = list(filter(None, executor.map(extract_embedding, data.images)))
    return {"embeddings": embeddings[:5]}

@app.post("/store-retrieve-embeddings")
async def store_embeddings(data: List[StoredEmbedding]):
    try:
        stored_embeddings.clear()
        for embedding_data in data:
            for emb in embedding_data.embedding:
                if np.array(emb).shape == (embedding_dimension,):
                    stored_embeddings.append({
                        "user_id": embedding_data.user_id,
                        "name": embedding_data.name,
                        "section": embedding_data.section,
                        "standard_division": embedding_data.standard_division,
                        "embedding": emb,
                    })
        return {
            "message": "Embeddings stored successfully",
            "count": len(stored_embeddings),
            "data": stored_embeddings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embedd-live-face")
async def embedd_live_face(data: LiveImageData):
    embedding = extract_embedding(data.image)
    if embedding is None:
        return {"error": "No face detected or embedding failed"}

    threshold = 75
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
            break

    if match_found:
        return {"status": "Match Found", "matches": matching_users}
    else:
        return {
            "status": "No Match Found",
            "error": "No match found",
            "confidence": "highest",
            "live_face_embedding": embedding
        }

@app.get("/")
def read_root():
    return {"message": "Face Recognition API is live!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
