import { Router as ExpressRouter } from 'express';
import { prisma } from '../index';
import { authenticate } from '../middleware/auth';
import { AppError } from '../middleware/errorHandler';
import { Router } from '../types/express';

const router: Router = ExpressRouter();

// Like a video
router.post('/:videoId', authenticate, async (req, res, next) => {
  try {
    const { videoId } = req.params;

    // Check if video exists
    const video = await prisma.video.findUnique({
      where: { id: videoId },
    });

    if (!video) {
      throw new AppError(404, 'Video not found');
    }

    // Check if already liked
    const existingLike = await prisma.like.findUnique({
      where: {
        userId_videoId: {
          userId: req.user!.id,
          videoId,
        },
      },
    });

    if (existingLike) {
      throw new AppError(400, 'Video already liked');
    }

    // Create like and update video like count
    await prisma.$transaction([
      prisma.like.create({
        data: {
          userId: req.user!.id,
          videoId,
        },
      }),
      prisma.video.update({
        where: { id: videoId },
        data: {
          likes: {
            increment: 1,
          },
        },
      }),
    ]);

    res.json({
      status: 'success',
      data: null,
    });
  } catch (error) {
    next(error);
  }
});

// Unlike a video
router.delete('/:videoId', authenticate, async (req, res, next) => {
  try {
    const { videoId } = req.params;

    // Delete like and update video like count
    await prisma.$transaction([
      prisma.like.delete({
        where: {
          userId_videoId: {
            userId: req.user!.id,
            videoId,
          },
        },
      }),
      prisma.video.update({
        where: { id: videoId },
        data: {
          likes: {
            decrement: 1,
          },
        },
      }),
    ]);

    res.json({
      status: 'success',
      data: null,
    });
  } catch (error) {
    next(error);
  }
});

// Get liked videos
router.get('/me', authenticate, async (req, res, next) => {
  try {
    const page = parseInt(req.query.page as string || '1');
    const limit = parseInt(req.query.limit as string || '10');
    const offset = (page - 1) * limit;

    const likes = await prisma.like.findMany({
      where: {
        userId: req.user!.id,
      },
      take: limit,
      skip: offset,
      orderBy: {
        createdAt: 'desc',
      },
      include: {
        video: {
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
          },
        },
      },
    });

    const total = await prisma.like.count({
      where: {
        userId: req.user!.id,
      },
    });

    res.json({
      status: 'success',
      data: {
        likes: likes.map((like: { video: any }) => like.video),
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      },
    });
  } catch (error) {
    next(error);
  }
});

export { router as likeRouter }; 