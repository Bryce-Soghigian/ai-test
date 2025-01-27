import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { PrismaClient } from '@prisma/client';
import Redis from 'ioredis';
import { MongoClient } from 'mongodb';
import { logger } from './utils/logger';
import { errorHandler } from './middleware/errorHandler';
import { authRouter } from './routes/auth';
import { userRouter } from './routes/user';
import { videoRouter } from './routes/video';
import { commentRouter } from './routes/comment';
import { likeRouter } from './routes/like';
import { soundRouter } from './routes/sound';
import { hashtagRouter } from './routes/hashtag';
import { recommendationRouter } from './routes/recommendation';
import { setupWebSocketHandlers } from './websocket';

// Initialize database clients
export const prisma = new PrismaClient();
export const redis = new Redis(process.env.REDIS_URL);
export const mongodb = new MongoClient(process.env.MONGODB_URL || '');

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(
  rateLimit({
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW || '60000'),
    max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100'),
  })
);

// Routes
app.use('/api/auth', authRouter);
app.use('/api/users', userRouter);
app.use('/api/videos', videoRouter);
app.use('/api/comments', commentRouter);
app.use('/api/likes', likeRouter);
app.use('/api/sounds', soundRouter);
app.use('/api/hashtags', hashtagRouter);
app.use('/api/recommendations', recommendationRouter);

// Error handling
app.use(errorHandler);

// WebSocket setup
setupWebSocketHandlers(wss);

const PORT = process.env.PORT || 3001;

async function bootstrap() {
  try {
    // Connect to MongoDB
    await mongodb.connect();
    logger.info('Connected to MongoDB');

    // Test Redis connection
    await redis.ping();
    logger.info('Connected to Redis');

    // Test Prisma connection
    await prisma.$connect();
    logger.info('Connected to PostgreSQL');

    server.listen(PORT, () => {
      logger.info(`Server is running on port ${PORT}`);
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

bootstrap();

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received. Shutting down gracefully...');
  
  server.close(() => {
    logger.info('HTTP server closed');
  });

  try {
    await Promise.all([
      prisma.$disconnect(),
      redis.quit(),
      mongodb.close(),
    ]);
    logger.info('Database connections closed');
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown:', error);
    process.exit(1);
  }
}); 