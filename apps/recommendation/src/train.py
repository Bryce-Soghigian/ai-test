import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import os

from .models import VideoEmbeddingModel, UserEmbeddingModel
from .config import Settings

settings = Settings()

# Load pre-trained text encoder
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

class VideoDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path) as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare text features
        text = f"{item['caption']} {' '.join(item['hashtags'])}"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embedding = text_encoder(**inputs).last_hidden_state.mean(dim=1)
        
        # Prepare numerical features
        numerical_features = torch.tensor([
            item["duration"],
            item["width"],
            item["height"],
            item["user_followers"],
            item["user_likes"],
        ])
        
        # Prepare target (engagement metrics)
        target = torch.tensor([
            item["views"],
            item["likes"],
            item["comments"],
            item["shares"],
            item["watch_time"],
            item["completion_rate"],
        ])
        
        return {
            "text_embedding": text_embedding.squeeze(0),
            "numerical_features": numerical_features,
            "target": target,
        }

class UserDataset(Dataset):
    def __init__(self, data_path, video_embeddings):
        self.data = []
        with open(data_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        self.video_embeddings = video_embeddings
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get average embeddings for watched and liked videos
        watched_embeddings = [
            self.video_embeddings[vid] for vid in item["watched_videos"]
            if vid in self.video_embeddings
        ]
        liked_embeddings = [
            self.video_embeddings[vid] for vid in item["liked_videos"]
            if vid in self.video_embeddings
        ]
        
        watched_avg = torch.tensor(np.mean(watched_embeddings, axis=0))
        liked_avg = torch.tensor(np.mean(liked_embeddings, axis=0))
        
        # Prepare numerical features
        numerical_features = torch.tensor([
            len(item["watched_videos"]),
            len(item["liked_videos"]),
            len(item["followed_users"]),
            item["average_watch_time"],
            item["completion_rate"],
        ])
        
        # Prepare target (future engagement metrics)
        target = torch.tensor([
            item["future_watch_time"],
            item["future_completion_rate"],
            item["future_like_rate"],
            item["future_comment_rate"],
            item["future_share_rate"],
        ])
        
        return {
            "watched_embedding": watched_avg,
            "liked_embedding": liked_avg,
            "numerical_features": numerical_features,
            "target": target,
        }

def train_video_model(train_path, val_path, epochs=10, batch_size=32):
    # Create datasets and dataloaders
    train_dataset = VideoDataset(train_path)
    val_dataset = VideoDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and optimizer
    model = VideoEmbeddingModel()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            optimizer.zero_grad()
            
            embedding = model(
                batch["text_embedding"],
                batch["numerical_features"],
            )
            
            # Simple prediction head for training
            pred = nn.Linear(settings.embedding_dim, 6)(embedding)
            loss = criterion(pred, batch["target"])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                embedding = model(
                    batch["text_embedding"],
                    batch["numerical_features"],
                )
                
                pred = nn.Linear(settings.embedding_dim, 6)(embedding)
                loss = criterion(pred, batch["target"])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(settings.model_path, "video_model.pt"))
    
    return model

def train_user_model(train_path, val_path, video_embeddings, epochs=10, batch_size=32):
    # Create datasets and dataloaders
    train_dataset = UserDataset(train_path, video_embeddings)
    val_dataset = UserDataset(val_path, video_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and optimizer
    model = UserEmbeddingModel()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            optimizer.zero_grad()
            
            embedding = model(
                batch["watched_embedding"],
                batch["liked_embedding"],
                batch["numerical_features"],
            )
            
            # Simple prediction head for training
            pred = nn.Linear(settings.embedding_dim, 5)(embedding)
            loss = criterion(pred, batch["target"])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                embedding = model(
                    batch["watched_embedding"],
                    batch["liked_embedding"],
                    batch["numerical_features"],
                )
                
                pred = nn.Linear(settings.embedding_dim, 5)(embedding)
                loss = criterion(pred, batch["target"])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(settings.model_path, "user_model.pt"))
    
    return model

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs(settings.model_path, exist_ok=True)
    
    # Train video model
    print("Training video model...")
    video_model = train_video_model(
        train_path="data/video_train.jsonl",
        val_path="data/video_val.jsonl",
    )
    
    # Generate video embeddings for user model training
    print("Generating video embeddings...")
    video_embeddings = {}
    video_dataset = VideoDataset("data/video_train.jsonl")
    for i in tqdm(range(len(video_dataset))):
        batch = video_dataset[i]
        with torch.no_grad():
            embedding = video_model(
                batch["text_embedding"].unsqueeze(0),
                batch["numerical_features"].unsqueeze(0),
            )
        video_embeddings[video_dataset.data[i]["id"]] = embedding.squeeze(0).numpy()
    
    # Train user model
    print("Training user model...")
    user_model = train_user_model(
        train_path="data/user_train.jsonl",
        val_path="data/user_val.jsonl",
        video_embeddings=video_embeddings,
    ) 