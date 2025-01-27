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
    fileSize: parseInt(process.env.MAX_VIDEO_SIZE_MB || '100') * 1024 * 1024, // MB to bytes
  },
});

const s3 = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
  },
});

const createVideoSchema = z.object({
  caption: z.string().min(1).max(300),
  soundId: z.string().optional(),
  hashtags: z.array(z.string()).optional(),
});

// Upload video
router.post('/', authenticate, upload.single('video'), async (req, res, next) => {
  try {
    const { caption, soundId, hashtags } = createVideoSchema.parse(req.body);
    const videoFile = req.file;

    if (!videoFile) {
      throw new AppError(400, 'No video file provided');
    }

    // Generate unique filename
    const filename = `${Date.now()}-${Math.random().toString(36).substring(2)}`;
    const videoKey = `videos/${filename}.mp4`;
    const thumbnailKey = `thumbnails/${filename}.jpg`;

    // Upload video to S3
    await s3.send(new PutObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: videoKey,
      Body: videoFile.buffer,
      ContentType: 'video/mp4',
    }));

    // Generate and upload thumbnail
    const thumbnail = await sharp(videoFile.buffer)
      .resize(720, 1280, { fit: 'contain' })
      .jpeg({ quality: 80 })
      .toBuffer();

    await s3.send(new PutObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: thumbnailKey,
      Body: thumbnail,
      ContentType: 'image/jpeg',
    }));

    // Create video record
    const video = await prisma.video.create({
      data: {
        caption,
        videoUrl: `https://${process.env.AWS_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${videoKey}`,
        thumbnailUrl: `https://${process.env.AWS_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${thumbnailKey}`,
        userId: req.user!.id,
        soundId,
        duration: 0, // TODO: Extract duration from video metadata
        width: 720,
        height: 1280,
        hashtags: hashtags ? {
          create: hashtags.map(name => ({
            hashtag: {
              connectOrCreate: {
                where: { name },
                create: { name },
              },
            },
          })),
        } : undefined,
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

    res.status(201).json({
      status: 'success',
      data: { video },
    });
  } catch (error) {
    next(error);
  }
});

// Get video feed
router.get('/feed', async (req, res, next) => {
  try {
    const page = parseInt(req.query.page as string || '1');
    const limit = parseInt(req.query.limit as string || '10');
    const offset = (page - 1) * limit;

    const videos = await prisma.video.findMany({
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

    const total = await prisma.video.count();

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

// Get video by ID
router.get('/:id', async (req, res, next) => {
  try {
    const video = await prisma.video.findUnique({
      where: { id: req.params.id },
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

    if (!video) {
      throw new AppError(404, 'Video not found');
    }

    res.json({
      status: 'success',
      data: { video },
    });
  } catch (error) {
    next(error);
  }
});

// Delete video
router.delete('/:id', authenticate, async (req, res, next) => {
  try {
    const video = await prisma.video.findUnique({
      where: { id: req.params.id },
    });

    if (!video) {
      throw new AppError(404, 'Video not found');
    }

    if (video.userId !== req.user!.id) {
      throw new AppError(403, 'Not authorized to delete this video');
    }

    await prisma.video.delete({
      where: { id: req.params.id },
    });

    // TODO: Delete video and thumbnail from S3

    res.json({
      status: 'success',
      data: null,
    });
  } catch (error) {
    next(error);
  }
});

export { router as videoRouter }; 