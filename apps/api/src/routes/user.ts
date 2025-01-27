import { Router as ExpressRouter } from 'express';
import { z } from 'zod';
import multer from 'multer';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import sharp from 'sharp';
import { prisma } from '../index';
import { authenticate } from '../middleware/auth';
import { AppError } from '../middleware/errorHandler';
import { Router } from '../types/express';

const router: Router = ExpressRouter();
const upload = multer({
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB
  },
});

const s3 = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
  },
});

const updateProfileSchema = z.object({
  username: z.string().min(3).max(20).optional(),
  name: z.string().optional(),
  bio: z.string().max(160).optional(),
});

// Get user profile
router.get('/:username', async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({
      where: { username: req.params.username },
      select: {
        id: true,
        username: true,
        name: true,
        bio: true,
        avatar: true,
        _count: {
          select: {
            followers: true,
            following: true,
            videos: true,
          },
        },
      },
    });

    if (!user) {
      throw new AppError(404, 'User not found');
    }

    res.json({
      status: 'success',
      data: { user },
    });
  } catch (error) {
    next(error);
  }
});

// Update profile
router.patch('/profile', authenticate, async (req, res, next) => {
  try {
    const { username, name, bio } = updateProfileSchema.parse(req.body);

    if (username) {
      // Check if username is taken
      const existingUser = await prisma.user.findUnique({
        where: { username },
      });

      if (existingUser && existingUser.id !== req.user!.id) {
        throw new AppError(409, 'Username is already taken');
      }
    }

    const user = await prisma.user.update({
      where: { id: req.user!.id },
      data: {
        username,
        name,
        bio,
      },
      select: {
        id: true,
        username: true,
        name: true,
        bio: true,
        avatar: true,
      },
    });

    res.json({
      status: 'success',
      data: { user },
    });
  } catch (error) {
    next(error);
  }
});

// Upload avatar
router.post('/avatar', authenticate, upload.single('avatar'), async (req, res, next) => {
  try {
    const avatarFile = req.file;

    if (!avatarFile) {
      throw new AppError(400, 'No avatar file provided');
    }

    // Process image
    const processedImage = await sharp(avatarFile.buffer)
      .resize(400, 400, { fit: 'cover' })
      .jpeg({ quality: 80 })
      .toBuffer();

    // Upload to S3
    const key = `avatars/${req.user!.id}-${Date.now()}.jpg`;
    await s3.send(new PutObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: key,
      Body: processedImage,
      ContentType: 'image/jpeg',
    }));

    // Update user record
    const user = await prisma.user.update({
      where: { id: req.user!.id },
      data: {
        avatar: `https://${process.env.AWS_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${key}`,
      },
      select: {
        id: true,
        username: true,
        name: true,
        bio: true,
        avatar: true,
      },
    });

    res.json({
      status: 'success',
      data: { user },
    });
  } catch (error) {
    next(error);
  }
});

// Follow user
router.post('/:id/follow', authenticate, async (req, res, next) => {
  try {
    const targetUserId = req.params.id;

    if (targetUserId === req.user!.id) {
      throw new AppError(400, 'Cannot follow yourself');
    }

    // Check if target user exists
    const targetUser = await prisma.user.findUnique({
      where: { id: targetUserId },
    });

    if (!targetUser) {
      throw new AppError(404, 'User not found');
    }

    // Create follow relationship
    await prisma.follow.create({
      data: {
        followerId: req.user!.id,
        followingId: targetUserId,
      },
    });

    res.json({
      status: 'success',
      data: null,
    });
  } catch (error) {
    next(error);
  }
});

// Unfollow user
router.delete('/:id/follow', authenticate, async (req, res, next) => {
  try {
    const targetUserId = req.params.id;

    await prisma.follow.delete({
      where: {
        followerId_followingId: {
          followerId: req.user!.id,
          followingId: targetUserId,
        },
      },
    });

    res.json({
      status: 'success',
      data: null,
    });
  } catch (error) {
    next(error);
  }
});

// Get user's videos
router.get('/:username/videos', async (req, res, next) => {
  try {
    const page = parseInt(req.query.page as string || '1');
    const limit = parseInt(req.query.limit as string || '10');
    const offset = (page - 1) * limit;

    const user = await prisma.user.findUnique({
      where: { username: req.params.username },
    });

    if (!user) {
      throw new AppError(404, 'User not found');
    }

    const videos = await prisma.video.findMany({
      where: { userId: user.id },
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
      where: { userId: user.id },
    });

    res.json({
      status: 'success',
      data: {
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

export { router as userRouter }; 