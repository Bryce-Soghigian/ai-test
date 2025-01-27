import { Router as ExpressRouter } from 'express';
import { prisma } from '../index';
import { AppError } from '../middleware/errorHandler';
import { Router } from '../types/express';

const router: Router = ExpressRouter();

// Search hashtags
router.get('/search', async (req, res, next) => {
  try {
    const query = req.query.q as string;
    const limit = parseInt(req.query.limit as string || '10');

    if (!query) {
      throw new AppError(400, 'Search query is required');
    }

    const hashtags = await prisma.hashtag.findMany({
      where: {
        name: {
          contains: query,
          mode: 'insensitive',
        },
      },
      take: limit,
      orderBy: [
        { videoCount: 'desc' },
        { viewCount: 'desc' },
      ],
    });

    res.json({
      status: 'success',
      data: { hashtags },
    });
  } catch (error) {
    next(error);
  }
});

// Get trending hashtags
router.get('/trending', async (req, res, next) => {
  try {
    const limit = parseInt(req.query.limit as string || '10');

    const hashtags = await prisma.hashtag.findMany({
      take: limit,
      orderBy: [
        { videoCount: 'desc' },
        { viewCount: 'desc' },
      ],
      where: {
        videoCount: {
          gt: 0,
        },
      },
    });

    res.json({
      status: 'success',
      data: { hashtags },
    });
  } catch (error) {
    next(error);
  }
});

// Get hashtag details and videos
router.get('/:name', async (req, res, next) => {
  try {
    const { name } = req.params;
    const page = parseInt(req.query.page as string || '1');
    const limit = parseInt(req.query.limit as string || '10');
    const offset = (page - 1) * limit;

    const hashtag = await prisma.hashtag.findUnique({
      where: { name },
    });

    if (!hashtag) {
      throw new AppError(404, 'Hashtag not found');
    }

    // Get videos with this hashtag
    const videos = await prisma.video.findMany({
      where: {
        hashtags: {
          some: {
            hashtagId: hashtag.id,
          },
        },
      },
      take: limit,
      skip: offset,
      orderBy: {
        createdAt: 'desc',
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
      },
    });

    const total = await prisma.video.count({
      where: {
        hashtags: {
          some: {
            hashtagId: hashtag.id,
          },
        },
      },
    });

    // Update view count
    await prisma.hashtag.update({
      where: { id: hashtag.id },
      data: {
        viewCount: {
          increment: 1,
        },
      },
    });

    res.json({
      status: 'success',
      data: {
        hashtag,
        videos,
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

export { router as hashtagRouter }; 