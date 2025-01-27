import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
import json
import os
from tqdm import tqdm
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel

from .models import VideoEmbeddingModel, UserEmbeddingModel
from .config import Settings
from .monitor import RecommendationMonitor

settings = Settings()
monitor = RecommendationMonitor()

class EngagementDataset(Dataset):
    def __init__(self, start_date: datetime, end_date: datetime):
        self.data = []
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        
        # Get engagement data from monitor
        metrics_path = "metrics/recommendation_metrics.csv"
        if os.path.exists(metrics_path):
            import pandas as pd
            df = pd.read_csv(metrics_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
            
            # Group by video_id and calculate engagement metrics
            engagement_data = df.groupby("video_id").agg({
                "watch_time": "mean",
                "completion_rate": "mean",
                "like_rate": "mean",
            }).reset_index()
            
            # Get video embeddings
            video_collection = Collection("video_embeddings")
            for _, row in engagement_data.iterrows():
                video_data = video_collection.query(
                    expr=f"video_id == '{row['video_id']}'",
                    output_fields=["embedding"],
                )
                if video_data:
                    self.data.append({
                        "video_id": row["video_id"],
                        "embedding": video_data[0]["embedding"],
                        "engagement": {
                            "watch_time": row["watch_time"],
                            "completion_rate": row["completion_rate"],
                            "like_rate": row["like_rate"],
                        },
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "embedding": torch.tensor(item["embedding"]),
            "target": torch.tensor([
                item["engagement"]["watch_time"],
                item["engagement"]["completion_rate"],
                item["engagement"]["like_rate"],
            ]),
        }

def retrain_models(days_of_data: int = 7):
    """Retrain models using recent engagement data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_of_data)
    
    print(f"Loading engagement data from {start_date} to {end_date}")
    dataset = EngagementDataset(start_date, end_date)
    
    if len(dataset) == 0:
        print("No engagement data available for retraining")
        return
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize models
    video_model = VideoEmbeddingModel()
    user_model = UserEmbeddingModel()
    
    # Load existing weights
    video_model.load_state_dict(torch.load(os.path.join(settings.model_path, "video_model.pt")))
    user_model.load_state_dict(torch.load(os.path.join(settings.model_path, "user_model.pt")))
    
    # Fine-tune models
    print("Fine-tuning models...")
    video_optimizer = optim.Adam(video_model.parameters(), lr=1e-4)
    user_optimizer = optim.Adam(user_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    for epoch in range(5):  # Fewer epochs for fine-tuning
        # Training
        video_model.train()
        user_model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/5 - Training"):
            video_optimizer.zero_grad()
            user_optimizer.zero_grad()
            
            # Forward pass
            video_embedding = video_model(batch["embedding"])
            pred = nn.Linear(settings.embedding_dim, 3)(video_embedding)
            loss = criterion(pred, batch["target"])
            
            # Backward pass
            loss.backward()
            video_optimizer.step()
            user_optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        video_model.eval()
        user_model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                video_embedding = video_model(batch["embedding"])
                pred = nn.Linear(settings.embedding_dim, 3)(video_embedding)
                loss = criterion(pred, batch["target"])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best models
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving improved models...")
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(video_model.state_dict(), 
                      os.path.join(settings.model_path, f"video_model_{timestamp}.pt"))
            torch.save(user_model.state_dict(), 
                      os.path.join(settings.model_path, f"user_model_{timestamp}.pt"))
            
            # Update current models
            torch.save(video_model.state_dict(), 
                      os.path.join(settings.model_path, "video_model.pt"))
            torch.save(user_model.state_dict(), 
                      os.path.join(settings.model_path, "user_model.pt"))
    
    print("Retraining complete!")
    
    # Log retraining metrics
    monitor.log_model_performance(
        embedding_type="retrain",
        inference_time=0.0,
        embedding_norm=best_val_loss,
        timestamp=datetime.now(),
    )

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs(settings.model_path, exist_ok=True)
    
    # Retrain models using last 7 days of data
    retrain_models(days_of_data=7) 