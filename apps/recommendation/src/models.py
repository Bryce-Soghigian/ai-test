import torch
import torch.nn as nn

from .config import Settings

settings = Settings()

class VideoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Text embedding projection
        self.text_projection = nn.Sequential(
            nn.Linear(settings.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        # Numerical features projection
        self.numerical_projection = nn.Sequential(
            nn.Linear(5, 64),  # 5 numerical features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )
        
        # Final projection to embedding space
        self.final_projection = nn.Sequential(
            nn.Linear(160, settings.embedding_dim),  # 128 + 32 = 160
            nn.LayerNorm(settings.embedding_dim),
        )
    
    def forward(self, text_embedding, numerical_features):
        # Project text embedding
        text_features = self.text_projection(text_embedding)
        
        # Project numerical features
        numerical_features = self.numerical_projection(numerical_features)
        
        # Concatenate features
        combined = torch.cat([text_features, numerical_features], dim=-1)
        
        # Final projection
        embedding = self.final_projection(combined)
        
        return embedding

class UserEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Watched videos embedding projection
        self.watched_projection = nn.Sequential(
            nn.Linear(settings.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        # Liked videos embedding projection
        self.liked_projection = nn.Sequential(
            nn.Linear(settings.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        # Numerical features projection
        self.numerical_projection = nn.Sequential(
            nn.Linear(5, 64),  # 5 numerical features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )
        
        # Final projection to embedding space
        self.final_projection = nn.Sequential(
            nn.Linear(288, settings.embedding_dim),  # 128 + 128 + 32 = 288
            nn.LayerNorm(settings.embedding_dim),
        )
    
    def forward(self, watched_embedding, liked_embedding, numerical_features):
        # Project watched videos embedding
        watched_features = self.watched_projection(watched_embedding)
        
        # Project liked videos embedding
        liked_features = self.liked_projection(liked_embedding)
        
        # Project numerical features
        numerical_features = self.numerical_projection(numerical_features)
        
        # Concatenate features
        combined = torch.cat([
            watched_features,
            liked_features,
            numerical_features,
        ], dim=-1)
        
        # Final projection
        embedding = self.final_projection(combined)
        
        return embedding 