import json
import random
import numpy as np
from faker import Faker
import os
from tqdm import tqdm

fake = Faker()

def generate_video_data(num_videos=1000):
    videos = []
    
    # Generate some hashtags to choose from
    hashtags = [
        "fyp", "foryou", "viral", "trending", "funny", "dance", "music",
        "comedy", "food", "fashion", "beauty", "fitness", "sports", "gaming",
        "art", "diy", "lifehacks", "tutorial", "cooking", "pets", "nature",
        "travel", "motivation", "business", "technology",
    ]
    
    for i in range(num_videos):
        # Generate random video data
        duration = random.uniform(5, 180)  # 5s to 3min
        width = random.choice([720, 1080])
        height = random.choice([1280, 1920])
        user_followers = int(np.random.lognormal(10, 2))
        user_likes = int(user_followers * random.uniform(0.1, 2))
        
        # Generate engagement metrics
        base_engagement = np.random.lognormal(10, 2)
        views = int(base_engagement)
        likes = int(views * random.uniform(0.1, 0.5))
        comments = int(likes * random.uniform(0.05, 0.2))
        shares = int(likes * random.uniform(0.01, 0.1))
        watch_time = duration * random.uniform(0.3, 1.0)
        completion_rate = watch_time / duration
        
        video = {
            "id": f"video_{i}",
            "caption": fake.text(max_nb_chars=150),
            "hashtags": random.sample(hashtags, random.randint(1, 5)),
            "duration": duration,
            "width": width,
            "height": height,
            "user_followers": user_followers,
            "user_likes": user_likes,
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "watch_time": watch_time,
            "completion_rate": completion_rate,
        }
        
        videos.append(video)
    
    return videos

def generate_user_data(num_users=1000, videos=None):
    users = []
    
    for i in range(num_users):
        # Generate random user data
        num_watched = random.randint(10, 100)
        watched_videos = random.sample(videos, num_watched)
        
        # Generate liked videos (subset of watched)
        num_liked = int(num_watched * random.uniform(0.1, 0.5))
        liked_videos = random.sample(watched_videos, num_liked)
        
        # Generate followed users
        num_followed = random.randint(5, 50)
        followed_users = [f"user_{j}" for j in range(num_followed)]
        
        # Generate engagement metrics
        average_watch_time = random.uniform(5, 60)
        completion_rate = random.uniform(0.3, 1.0)
        
        # Generate future engagement metrics
        future_watch_time = average_watch_time * random.uniform(0.8, 1.2)
        future_completion_rate = completion_rate * random.uniform(0.8, 1.2)
        future_like_rate = random.uniform(0.1, 0.5)
        future_comment_rate = random.uniform(0.01, 0.1)
        future_share_rate = random.uniform(0.001, 0.05)
        
        user = {
            "id": f"user_{i}",
            "watched_videos": [v["id"] for v in watched_videos],
            "liked_videos": [v["id"] for v in liked_videos],
            "followed_users": followed_users,
            "average_watch_time": average_watch_time,
            "completion_rate": completion_rate,
            "future_watch_time": future_watch_time,
            "future_completion_rate": future_completion_rate,
            "future_like_rate": future_like_rate,
            "future_comment_rate": future_comment_rate,
            "future_share_rate": future_share_rate,
        }
        
        users.append(user)
    
    return users

def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate video data
    print("Generating video data...")
    videos = generate_video_data(num_videos=10000)
    
    # Split into train/val
    train_videos = videos[:8000]
    val_videos = videos[8000:]
    
    save_jsonl(train_videos, "data/video_train.jsonl")
    save_jsonl(val_videos, "data/video_val.jsonl")
    
    # Generate user data
    print("Generating user data...")
    users = generate_user_data(num_users=5000, videos=train_videos)
    
    # Split into train/val
    train_users = users[:4000]
    val_users = users[4000:]
    
    save_jsonl(train_users, "data/user_train.jsonl")
    save_jsonl(val_users, "data/user_val.jsonl") 