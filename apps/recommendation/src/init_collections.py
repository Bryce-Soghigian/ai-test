from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from .config import Settings

settings = Settings()

def init_collections():
    # Connect to Milvus
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )

    # Create video embeddings collection
    if not utility.has_collection("video_embeddings"):
        video_fields = [
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        video_schema = CollectionSchema(
            fields=video_fields,
            description="Video embeddings for recommendation",
        )
        video_collection = Collection(
            name="video_embeddings",
            schema=video_schema,
            using="default",
        )

        # Create index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        video_collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        print("Created video embeddings collection")
    else:
        print("Video embeddings collection already exists")

    # Create user embeddings collection
    if not utility.has_collection("user_embeddings"):
        user_fields = [
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            FieldSchema(name="updated_at", dtype=DataType.INT64),
        ]
        user_schema = CollectionSchema(
            fields=user_fields,
            description="User embeddings for recommendation",
        )
        user_collection = Collection(
            name="user_embeddings",
            schema=user_schema,
            using="default",
        )

        # Create index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        user_collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        print("Created user embeddings collection")
    else:
        print("User embeddings collection already exists")

if __name__ == "__main__":
    init_collections() 