import { Router as ExpressRouter } from 'express';
import { z } from 'zod';
import { prisma } from '../index';
import { authenticate } from '../middleware/auth';
import { AppError } from '../middleware/errorHandler';
import { Router } from '../types/express';

const router: Router = ExpressRouter();

const createCommentSchema = z.object({
  text: z.string().min(1).max(300),
});

// Create comment
router.post('/:videoId', authenticate, async (req, res, next) => {
  try {
    const { videoId } = req.params;
    const { text } = createCommentSchema.parse(req.body);

    // Check if video exists
    const video = await prisma.video.findUnique({
      where: { id: videoId },
    });

    if (!video) {
      throw new AppError(404, 'Video not found');
    }

    // Create comment and update video comment count
    const [comment] = await prisma.$transaction([
      prisma.comment.create({
        data: {
          text,
          userId: req.user!.id,
          videoId,
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
        },
      }),
      prisma.video.update({
        where: { id: videoId },
        data: {
          comments: {
            increment: 1,
          },
        },
      }),
    ]);

    res.status(201).json({
      status: 'success',
      data: { comment },
    });
  } catch (error) {
    next(error);
  }
});

// Get video comments
router.get('/:videoId', async (req, res, next) => {
  try {
    const { videoId } = req.params;
    const page = parseInt(req.query.page as string || '1');
    const limit = parseInt(req.query.limit as string || '20');
    const offset = (page - 1) * limit;

    // Check if video exists
    const video = await prisma.video.findUnique({
      where: { id: videoId },
    });

    if (!video) {
      throw new AppError(404, 'Video not found');
    }

    const comments = await prisma.comment.findMany({
      where: { videoId },
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
      },
    });

    const total = await prisma.comment.count({
      where: { videoId },
    });

    res.json({
      status: 'success',
      data: {
        comments,
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

// Delete comment
router.delete('/:id', authenticate, async (req, res, next) => {
  try {
    const { id } = req.params;

    const comment = await prisma.comment.findUnique({
      where: { id },
      include: {
        video: true,
      },
    });

    if (!comment) {
      throw new AppError(404, 'Comment not found');
    }

    // Check if user owns the comment or the video
    if (comment.userId !== req.user!.id && comment.video.userId !== req.user!.id) {
      throw new AppError(403, 'Not authorized to delete this comment');
    }

    // Delete comment and update video comment count
    await prisma.$transaction([
      prisma.comment.delete({
        where: { id },
      }),
      prisma.video.update({
        where: { id: comment.videoId },
        data: {
          comments: {
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

export { router as commentRouter }; 