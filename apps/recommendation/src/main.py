from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from pymilvus import connections, Collection
import torch
from transformers import AutoTokenizer, AutoModel
import time
from datetime import datetime

from .models import VideoEmbeddingModel, UserEmbeddingModel
from .config import Settings
from .init_collections import init_collections
from .monitor import RecommendationMonitor

app = FastAPI(title="TikTok Clone Recommendation Service")
settings = Settings()
monitor = RecommendationMonitor()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    # Initialize Milvus collections
    init_collections()

    # Connect to Milvus
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )

    # Load models
    global tokenizer, text_encoder, video_model, user_model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    video_model = VideoEmbeddingModel()
    user_model = UserEmbeddingModel()

    # Load model weights
    video_model.load_state_dict(torch.load("models/video_model.pt"))
    user_model.load_state_dict(torch.load("models/user_model.pt"))
    video_model.eval()
    user_model.eval()

class VideoFeatures(BaseModel):
    caption: str
    hashtags: List[str]
    duration: float
    width: int
    height: int
    user_followers: int
    user_likes: int

class UserFeatures(BaseModel):
    watched_videos: List[str]
    liked_videos: List[str]
    followed_users: List[str]
    average_watch_time: float
    completion_rate: float

class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = 20
    cursor: Optional[str] = None

class VideoRecommendation(BaseModel):
    video_id: str
    score: float
    reason: str

class RecommendationResponse(BaseModel):
    items: List[VideoRecommendation]
    has_more: bool
    next_cursor: Optional[str]

@app.post("/embeddings/video")
async def create_video_embedding(video_id: str, features: VideoFeatures):
    try:
        start_time = time.time()

        # Encode text features
        text = f"{features.caption} {' '.join(features.hashtags)}"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embedding = text_encoder(**inputs).last_hidden_state.mean(dim=1)

        # Create numerical features
        numerical_features = torch.tensor([
            features.duration,
            features.width,
            features.height,
            features.user_followers,
            features.user_likes,
        ])

        # Generate video embedding
        with torch.no_grad():
            embedding = video_model(text_embedding, numerical_features)

        # Store in Milvus
        collection = Collection("video_embeddings")
        collection.insert([
            [video_id],  # Primary key
            embedding.numpy().tolist(),  # Vector field
            [int(time.time() * 1000)],  # Created at
        ])

        # Log metrics
        inference_time = time.time() - start_time
        monitor.log_model_performance(
            embedding_type="video",
            inference_time=inference_time,
            embedding_norm=float(torch.norm(embedding).item()),
            timestamp=datetime.now(),
        )

        return {"status": "success"}
    except Exception as e:
        monitor.log_system_health(
            response_time=time.time() - start_time,
            error_count=1,
            timestamp=datetime.now(),
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/user")
async def create_user_embedding(user_id: str, features: UserFeatures):
    try:
        start_time = time.time()

        # Get video embeddings for watched and liked videos
        collection = Collection("video_embeddings")
        watched_embeddings = collection.query(
            expr=f"video_id in {features.watched_videos}",
            output_fields=["embedding"],
        )
        liked_embeddings = collection.query(
            expr=f"video_id in {features.liked_videos}",
            output_fields=["embedding"],
        )

        # Average embeddings
        watched_avg = np.mean([e["embedding"] for e in watched_embeddings], axis=0)
        liked_avg = np.mean([e["embedding"] for e in liked_embeddings], axis=0)

        # Create numerical features
        numerical_features = torch.tensor([
            len(features.watched_videos),
            len(features.liked_videos),
            len(features.followed_users),
            features.average_watch_time,
            features.completion_rate,
        ])

        # Generate user embedding
        with torch.no_grad():
            embedding = user_model(
                torch.tensor(watched_avg),
                torch.tensor(liked_avg),
                numerical_features,
            )

        # Store in Milvus
        collection = Collection("user_embeddings")
        collection.insert([
            [user_id],  # Primary key
            embedding.numpy().tolist(),  # Vector field
            [int(time.time() * 1000)],  # Updated at
        ])

        # Log metrics
        inference_time = time.time() - start_time
        monitor.log_model_performance(
            embedding_type="user",
            inference_time=inference_time,
            embedding_norm=float(torch.norm(embedding).item()),
            timestamp=datetime.now(),
        )

        return {"status": "success"}
    except Exception as e:
        monitor.log_system_health(
            response_time=time.time() - start_time,
            error_count=1,
            timestamp=datetime.now(),
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        start_time = time.time()

        # Get user embedding
        user_collection = Collection("user_embeddings")
        user_results = user_collection.query(
            expr=f"user_id == {request.user_id}",
            output_fields=["embedding"],
        )

        if not user_results:
            raise HTTPException(status_code=404, detail="User embedding not found")

        user_embedding = user_results[0]["embedding"]

        # Search for similar videos
        video_collection = Collection("video_embeddings")
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        results = video_collection.search(
            data=[user_embedding],
            anns_field="embedding",
            param=search_params,
            limit=request.limit + 1,  # Get one extra to check if there are more
            expr=f"video_id > '{request.cursor}'" if request.cursor else None,
        )

        # Format results
        items = []
        for hit in results[0][:request.limit]:
            items.append(VideoRecommendation(
                video_id=hit.id,
                score=float(hit.distance),
                reason="Based on your watch history",
            ))

        has_more = len(results[0]) > request.limit
        next_cursor = results[0][request.limit - 1].id if has_more else None

        # Log metrics
        response_time = time.time() - start_time
        monitor.log_system_health(
            response_time=response_time,
            error_count=0,
            timestamp=datetime.now(),
        )

        return RecommendationResponse(
            items=items,
            has_more=has_more,
            next_cursor=next_cursor,
        )
    except Exception as e:
        monitor.log_system_health(
            response_time=time.time() - start_time,
            error_count=1,
            timestamp=datetime.now(),
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/{video_id}/engage")
async def record_engagement(
    video_id: str,
    watch_time: float,
    completion_rate: float,
    liked: bool,
    user_id: str,
):
    """Record user engagement with a recommended video."""
    try:
        monitor.log_engagement(
            user_id=user_id,
            video_id=video_id,
            watch_time=watch_time,
            completion_rate=completion_rate,
            liked=liked,
            timestamp=datetime.now(),
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get current recommendation system metrics."""
    try:
        metrics = monitor.get_metrics_summary()
        embedding_stats = monitor.analyze_embedding_distribution()
        return {
            "status": "success",
            "data": {
                "metrics": metrics,
                "embedding_stats": embedding_stats,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002) 