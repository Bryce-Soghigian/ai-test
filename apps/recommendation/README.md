# TikTok Clone Recommendation Service

This service provides personalized video recommendations for the TikTok clone application using deep learning and vector similarity search.

## Features

- Video embedding generation based on content and metadata
- User embedding generation based on watch history and engagement
- Personalized video recommendations using vector similarity search
- Support for pagination and cursor-based navigation
- Real-time recommendation updates based on user interactions

## Architecture

The recommendation system uses a two-tower neural network architecture:

1. **Video Tower**: Processes video content (captions, hashtags) and metadata (duration, dimensions, creator stats) to generate video embeddings.
2. **User Tower**: Processes user history (watched videos, liked videos) and engagement metrics to generate user embeddings.

Both towers project their inputs into the same embedding space, allowing for efficient similarity search using Milvus.

## Prerequisites

- Python 3.11+
- PyTorch
- FastAPI
- Milvus vector database
- Docker (for containerized deployment)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate sample training data:
```bash
python -m src.generate_data
```

3. Train the models:
```bash
python -m src.train
```

4. Start the service:
```bash
python -m src.main
```

Or using Docker:
```bash
docker build -t tiktok-recommendation .
docker run -p 3002:3002 tiktok-recommendation
```

## API Endpoints

### POST /embeddings/video
Generate and store embeddings for a new video.

Request:
```json
{
  "video_id": "string",
  "caption": "string",
  "hashtags": ["string"],
  "duration": 0,
  "width": 0,
  "height": 0,
  "user_followers": 0,
  "user_likes": 0
}
```

### POST /embeddings/user
Generate and store embeddings for a user based on their activity.

Request:
```json
{
  "user_id": "string",
  "watched_videos": ["string"],
  "liked_videos": ["string"],
  "followed_users": ["string"],
  "average_watch_time": 0,
  "completion_rate": 0
}
```

### POST /recommendations
Get personalized video recommendations for a user.

Request:
```json
{
  "user_id": "string",
  "limit": 20,
  "cursor": "string"
}
```

Response:
```json
{
  "items": [
    {
      "video_id": "string",
      "score": 0,
      "reason": "string"
    }
  ],
  "has_more": true,
  "next_cursor": "string"
}
```

## Model Training

The models are trained using a combination of content-based features and engagement metrics:

1. Video features:
   - Text features from captions and hashtags (using BERT embeddings)
   - Video metadata (duration, dimensions)
   - Creator statistics (followers, likes)
   - Engagement metrics (views, likes, comments, shares)

2. User features:
   - Average embeddings of watched videos
   - Average embeddings of liked videos
   - Engagement metrics (watch time, completion rate)
   - Social graph features (number of followed users)

The training process uses MSE loss to predict future engagement metrics, which helps learn embeddings that capture user preferences and video quality.

## Performance Optimization

1. Batch processing for embedding generation
2. Caching of frequently accessed embeddings
3. Efficient vector similarity search using Milvus
4. Model quantization for reduced memory usage
5. Asynchronous API endpoints for better concurrency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 