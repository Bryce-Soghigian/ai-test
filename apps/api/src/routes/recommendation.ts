import { Router as ExpressRouter } from 'express';
import { z } from 'zod';
import { prisma } from '../index';
import { redis } from '../index';
import { authenticate } from '../middleware/auth';
import { AppError } from '../middleware/errorHandler';
import { Router } from '../types/express';

const router: Router = ExpressRouter();

const getFeedSchema = z.object({
  limit: z.number().min(1).max(50).default(10),
  cursor: z.string().optional(),
});

// Get personalized feed
router.get('/feed', authenticate, async (req, res, next) => {
  try {
    const { limit, cursor } = getFeedSchema.parse({
      limit: parseInt(req.query.limit as string || '10'),
      cursor: req.query.cursor,
    });

    // Get user's recent interactions from Redis
    const recentInteractions = await redis.lrange(`user:${req.user!.id}:interactions`, 0, 100);
    
    // Get user's followed creators
    const following = await prisma.follow.findMany({
      where: { followerId: req.user!.id },
      select: { followingId: true },
    });
    const followingIds = following.map((f: { followingId: string }) => f.followingId);

    // Build query
    const query = {
      where: {
        OR: [
          // Videos from followed creators (higher weight)
          ...(followingIds.length > 0 ? [{
            userId: {
              in: followingIds,
            },
          }] : []),
          // Videos with hashtags user has interacted with
          {
            hashtags: {
              some: {
                hashtag: {
                  videos: {
                    some: {
                      id: {
                        in: recentInteractions,
                      },
                    },
                  },
                },
              },
            },
          },
          // Videos with sounds user has interacted with
          {
            soundId: {
              in: await prisma.video.findMany({
                where: { id: { in: recentInteractions } },
                select: { soundId: true },
              }).then((videos: Array<{ soundId: string | null }>) => 
                videos.map(v => v.soundId).filter(Boolean) as string[]
              ),
            },
          },
        ],
        // Don't show videos user has already seen
        NOT: {
          engagements: {
            some: {
              userId: req.user!.id,
            },
          },
        },
      },
      take: limit + 1, // Get one extra to check if there are more
      orderBy: [
        { createdAt: 'desc' },
        { likes: 'desc' },
      ],
      include: {
        user: {
          select: {
            id: true,
            username: true,
            name: true,
            avatar: true,
          },
        },
        hashtags: {
          include: {
            hashtag: true,
          },
        },
        sound: true,
      },
      cursor: cursor ? { id: cursor } : undefined,
    };

    // Get videos
    const videos = await prisma.video.findMany(query);

    // Check if there are more videos
    const hasMore = videos.length > limit;
    const nextCursor = hasMore ? videos[limit - 1].id : undefined;
    const items = hasMore ? videos.slice(0, -1) : videos;

    res.json({
      status: 'success',
      data: {
        items,
        hasMore,
        nextCursor,
      },
    });
  } catch (error) {
    next(error);
  }
});

// Get trending videos
router.get('/trending', async (req, res, next) => {
  try {
    const limit = parseInt(req.query.limit as string || '10');

    const videos = await prisma.video.findMany({
      take: limit,
      orderBy: [
        { likes: 'desc' },
        { comments: 'desc' },
        { shares: 'desc' },
      ],
      where: {
        createdAt: {
          gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // Last 7 days
        },
      },
      include: {
        user: {
          select: {
            id: true,
            username: true,
            name: true,
            avatar: true,
          },
        },
        hashtags: {
          include: {
            hashtag: true,
          },
        },
        sound: true,
      },
    });

    res.json({
      status: 'success',
      data: { videos },
    });
  } catch (error) {
    next(error);
  }
});

// Record video engagement
router.post('/:videoId/engage', authenticate, async (req, res, next) => {
  try {
    const { videoId } = req.params;
    const { watchTimeMs, completionRate } = z.object({
      watchTimeMs: z.number().min(0),
      completionRate: z.number().min(0).max(1),
    }).parse(req.body);

    // Record engagement
    await prisma.videoEngagement.create({
      data: {
        userId: req.user!.id,
        videoId,
        watchTimeMs,
        completionRate,
      },
    });

    // Add to user's recent interactions
    await redis.lpush(`user:${req.user!.id}:interactions`, videoId);
    await redis.ltrim(`user:${req.user!.id}:interactions`, 0, 999); // Keep last 1000 interactions

    res.json({
      status: 'success',
      data: null,
    });
  } catch (error) {
    next(error);
  }
});

export { router as recommendationRouter }; 