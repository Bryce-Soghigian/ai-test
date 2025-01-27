from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    model_path: str = "models"
    embedding_dim: int = 384  # Matches the MiniLM model output dimension
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8" 