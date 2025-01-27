export interface User {
  id: string;
  username: string;
  email: string;
  name?: string;
  bio?: string;
  avatar?: string;
  createdAt: Date;
  updatedAt: Date;
  followersCount: number;
  followingCount: number;
  likesCount: number;
}

export interface Video {
  id: string;
  userId: string;
  caption: string;
  videoUrl: string;
  thumbnailUrl: string;
  soundId?: string;
  duration: number;
  width: number;
  height: number;
  likes: number;
  shares: number;
  comments: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface Comment {
  id: string;
  videoId: string;
  userId: string;
  text: string;
  likes: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface Sound {
  id: string;
  name: string;
  artistName?: string;
  duration: number;
  audioUrl: string;
  usageCount: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface Hashtag {
  id: string;
  name: string;
  videoCount: number;
  viewCount: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface VideoEngagement {
  userId: string;
  videoId: string;
  watchTimeMs: number;
  completionRate: number;
  liked: boolean;
  shared: boolean;
  createdAt: Date;
}

// Recommendation types
export interface VideoEmbedding {
  videoId: string;
  embedding: number[];
  updatedAt: Date;
}

export interface UserEmbedding {
  userId: string;
  embedding: number[];
  updatedAt: Date;
}

export interface RecommendationResponse {
  videos: Video[];
  score: number;
  reason: string;
} 