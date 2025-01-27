# TikTok Clone with AI Recommendations

A modern short-form video platform built with Next.js, Node.js, and PyTorch.

## Project Structure

```
├── apps/
│   ├── web/                 # Next.js frontend
│   ├── api/                 # Main API service
│   ├── recommendation/      # ML recommendation service
│   └── worker/             # Background job processor
├── packages/
│   ├── database/           # Database schemas and migrations
│   ├── shared/             # Shared TypeScript types and utilities
│   └── config/             # Shared configuration
├── docker/                 # Docker configurations
└── k8s/                   # Kubernetes manifests
```

## Tech Stack

- **Frontend**: Next.js 14 (React) with TypeScript
- **Backend**: Node.js with Express and TypeScript
- **ML Service**: Python with FastAPI, PyTorch
- **Databases**: 
  - PostgreSQL (user data, relationships)
  - Redis (caching, real-time features)
  - MongoDB (video metadata, engagement data)
  - Milvus (vector store for embeddings)

## Getting Started

1. Prerequisites:
   - Node.js 18+
   - Python 3.9+
   - Docker
   - pnpm (recommended)

2. Installation:
   ```bash
   # Install dependencies
   pnpm install

   # Set up environment variables
   cp .env.example .env

   # Start development environment
   pnpm dev
   ```

## Development

- `pnpm dev`: Start all services in development mode
- `pnpm build`: Build all packages and apps
- `pnpm test`: Run tests
- `pnpm lint`: Run linting

## Architecture

The platform is built using a microservices architecture:

1. **Web Frontend**: Next.js application for the user interface
2. **API Service**: Main backend service handling user data and video operations
3. **Recommendation Service**: ML-powered content recommendation system
4. **Worker Service**: Background job processing (video transcoding, notifications)

## License

MIT 