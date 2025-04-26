import os
import cv2
import base64
import torch
import logging
import numpy as np
import insightface

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Suppress unnecessary logs
logging.getLogger("insightface").setLevel(logging.ERROR)
cv2.setLogLevel(0)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Environment setup
ctx_id = 0 if torch.cuda.is_available() else -1
embedding_dimension = 512  # Face embedding dimension

# Load models
from insightface.model_zoo import get_model
detector = get_model('scrfd_2.5g_bnkps')
recognizer = get_model('glintr100@insightface')
detector.prepare(ctx_id=ctx_id)
recognizer.prepare(ctx_id=ctx_id)

# FastAPI setup
app = FastAPI()

# Data models
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

# In-memory storage
stored_embeddings = []

# Utility functions
def decode_base64_image(base64_str: str):
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(img, (640, 640)) if img is not None else None
    except Exception:
        return None

def extract_face_embedding(base64_str: str):
    img = decode_base64_image(base64_str)
    if img is None:
        return None

    bboxes, landmarks = detector.detect(img, max_num=1, metric='default')
    if bboxes.shape[0] == 0:
        return None

    x1, y1, x2, y2, _ = bboxes[0].astype(int)
    face = img[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    embedding = recognizer.get(face).tolist()

    if np.array(embedding).shape == (embedding_dimension,):
        return embedding
    return None

def compute_cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return (dot / norm + 1) / 2 * 100  # 0 to 100%

# API Endpoints
@app.post("/extract-embedding")
async def extract_embeddings(data: ImageData):
    with ThreadPoolExecutor() as executor:
        embeddings = list(filter(None, executor.map(extract_face_embedding, data.images)))
    return {"embeddings": embeddings[:5]}

@app.post("/store-retrieve-embeddings")
async def store_embeddings(data: List[StoredEmbedding]):
    try:
        stored_embeddings.clear()
        for user_data in data:
            for emb in user_data.embedding:
                if np.array(emb).shape == (embedding_dimension,):
                    stored_embeddings.append({
                        "user_id": user_data.user_id,
                        "name": user_data.name,
                        "section": user_data.section,
                        "standard_division": user_data.standard_division,
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
    embedding = extract_face_embedding(data.image)
    if embedding is None:
        return {"error": "No face detected or embedding failed"}

    threshold = 75
    match_found = False
    matches = []

    for record in stored_embeddings:
        confidence = compute_cosine_similarity(record["embedding"], embedding)
        if confidence > threshold:
            matches.append({
                "user_id": record["user_id"],
                "name": record["name"],
                "section": record["section"],
                "standard_division": record["standard_division"],
                "confidence": confidence,
                "live_face_embedding": embedding
            })
            match_found = True
            break

    if match_found:
        return {"status": "Match Found", "matches": matches}
    else:
        return {
            "status": "No Match Found",
            "error": "No matching user found",
            "confidence": "highest",
            "live_face_embedding": embedding
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# Command to run:
# uvicorn main:app --host 0.0.0.0 --port=8000 --reload
