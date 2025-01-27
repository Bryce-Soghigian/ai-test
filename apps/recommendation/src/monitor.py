import numpy as np
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import ndcg_score, precision_score, recall_score
import torch
from pymilvus import connections, Collection

from .config import Settings

settings = Settings()

class RecommendationMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.window_size = timedelta(hours=1)
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
    
    def log_recommendation(
        self,
        user_id: str,
        recommended_videos: List[Tuple[str, float]],
        interacted_videos: List[str],
        timestamp: datetime,
    ):
        """Log a recommendation event and its outcomes."""
        # Calculate ranking metrics
        relevance = [1 if vid in interacted_videos else 0 for vid, _ in recommended_videos]
        if sum(relevance) > 0:  # Only calculate if there were any relevant items
            ndcg = ndcg_score([relevance], [[1] * len(relevance)])
            precision = precision_score([1 if vid in interacted_videos else 0 for vid, _ in recommended_videos[:10]], [1] * 10, zero_division=0)
            recall = recall_score([1 if vid in interacted_videos else 0 for vid, _ in recommended_videos], [1] * len(recommended_videos), zero_division=0)
        else:
            ndcg = precision = recall = 0.0

        # Log metrics
        self.metrics["ndcg"].append((timestamp, ndcg))
        self.metrics["precision@10"].append((timestamp, precision))
        self.metrics["recall"].append((timestamp, recall))
        
        # Clean up old metrics
        self._cleanup_old_metrics(timestamp)
    
    def log_engagement(
        self,
        user_id: str,
        video_id: str,
        watch_time: float,
        completion_rate: float,
        liked: bool,
        timestamp: datetime,
    ):
        """Log user engagement with a recommended video."""
        self.metrics["watch_time"].append((timestamp, watch_time))
        self.metrics["completion_rate"].append((timestamp, completion_rate))
        self.metrics["like_rate"].append((timestamp, 1 if liked else 0))
        
        # Clean up old metrics
        self._cleanup_old_metrics(timestamp)
    
    def log_model_performance(
        self,
        embedding_type: str,
        inference_time: float,
        embedding_norm: float,
        timestamp: datetime,
    ):
        """Log model performance metrics."""
        self.metrics[f"{embedding_type}_inference_time"].append((timestamp, inference_time))
        self.metrics[f"{embedding_type}_embedding_norm"].append((timestamp, embedding_norm))
        
        # Clean up old metrics
        self._cleanup_old_metrics(timestamp)
    
    def log_system_health(
        self,
        response_time: float,
        error_count: int,
        timestamp: datetime,
    ):
        """Log system health metrics."""
        self.metrics["response_time"].append((timestamp, response_time))
        self.metrics["error_count"].append((timestamp, error_count))
        
        # Clean up old metrics
        self._cleanup_old_metrics(timestamp)
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics in the current window."""
        now = datetime.now()
        summary = {}
        
        for metric_name, values in self.metrics.items():
            # Filter values in current window
            recent_values = [v for t, v in values if now - t <= self.window_size]
            
            if recent_values:
                summary[f"{metric_name}_mean"] = np.mean(recent_values)
                summary[f"{metric_name}_std"] = np.std(recent_values)
                summary[f"{metric_name}_min"] = np.min(recent_values)
                summary[f"{metric_name}_max"] = np.max(recent_values)
        
        return summary
    
    def analyze_embedding_distribution(self):
        """Analyze the distribution of embeddings in the vector space."""
        # Get video embeddings
        video_collection = Collection("video_embeddings")
        video_embeddings = video_collection.query(
            expr="",
            output_fields=["embedding"],
            limit=1000,  # Sample size
        )
        
        # Get user embeddings
        user_collection = Collection("user_embeddings")
        user_embeddings = user_collection.query(
            expr="",
            output_fields=["embedding"],
            limit=1000,  # Sample size
        )
        
        # Convert to numpy arrays
        video_vectors = np.array([e["embedding"] for e in video_embeddings])
        user_vectors = np.array([e["embedding"] for e in user_embeddings])
        
        # Calculate statistics
        stats = {
            "video_embedding_mean_norm": np.mean(np.linalg.norm(video_vectors, axis=1)),
            "user_embedding_mean_norm": np.mean(np.linalg.norm(user_vectors, axis=1)),
            "video_embedding_std": np.std(video_vectors, axis=0).mean(),
            "user_embedding_std": np.std(user_vectors, axis=0).mean(),
            "cosine_similarity_mean": np.mean(video_vectors @ user_vectors.T),
        }
        
        return stats
    
    def save_metrics(self, path: str):
        """Save metrics to a file."""
        # Convert metrics to DataFrame
        data = []
        for metric_name, values in self.metrics.items():
            for timestamp, value in values:
                data.append({
                    "metric": metric_name,
                    "timestamp": timestamp.isoformat(),
                    "value": value,
                })
        
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
    
    def _cleanup_old_metrics(self, current_time: datetime):
        """Remove metrics older than the window size."""
        for metric_name in list(self.metrics.keys()):
            self.metrics[metric_name] = [
                (t, v) for t, v in self.metrics[metric_name]
                if current_time - t <= self.window_size
            ]

# Example usage
if __name__ == "__main__":
    monitor = RecommendationMonitor()
    
    # Simulate some recommendations and engagements
    now = datetime.now()
    
    # Log recommendations
    monitor.log_recommendation(
        user_id="user_1",
        recommended_videos=[("video_1", 0.9), ("video_2", 0.8)],
        interacted_videos=["video_1"],
        timestamp=now,
    )
    
    # Log engagement
    monitor.log_engagement(
        user_id="user_1",
        video_id="video_1",
        watch_time=45.0,
        completion_rate=0.75,
        liked=True,
        timestamp=now,
    )
    
    # Log model performance
    monitor.log_model_performance(
        embedding_type="video",
        inference_time=0.05,
        embedding_norm=1.0,
        timestamp=now,
    )
    
    # Get metrics summary
    summary = monitor.get_metrics_summary()
    print("Metrics Summary:")
    print(json.dumps(summary, indent=2))
    
    # Analyze embedding distribution
    stats = monitor.analyze_embedding_distribution()
    print("\nEmbedding Distribution Analysis:")
    print(json.dumps(stats, indent=2))
    
    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    monitor.save_metrics("metrics/recommendation_metrics.csv") 