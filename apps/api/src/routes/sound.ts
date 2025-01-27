import { Router as ExpressRouter } from 'express';
import { z } from 'zod';
import multer from 'multer';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { prisma } from '../index';
import { authenticate } from '../middleware/auth';
import { AppError } from '../middleware/errorHandler';
import { Router } from '../types/express';

const router: Router = ExpressRouter();
const upload = multer({
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
  },
});

const s3 = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
  },
});

const createSoundSchema = z.object({
  name: z.string().min(1).max(100),
  artistName: z.string().optional(),
  duration: z.number().min(0),
});

// Upload sound
router.post('/', authenticate, upload.single('audio'), async (req, res, next) => {
  try {
    const { name, artistName, duration } = createSoundSchema.parse(req.body);
    const audioFile = req.file;

    if (!audioFile) {
      throw new AppError(400, 'No audio file provided');
    }

    // Generate unique filename
    const filename = `${Date.now()}-${Math.random().toString(36).substring(2)}`;
    const audioKey = `sounds/${filename}.mp3`;

    // Upload audio to S3
    await s3.send(new PutObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: audioKey,
      Body: audioFile.buffer,
      ContentType: 'audio/mpeg',
    }));

    // Create sound record
    const sound = await prisma.sound.create({
      data: {
        name,
        artistName,
        duration,
        audioUrl: `https://${process.env.AWS_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${audioKey}`,
      },
    });

    res.status(201).json({
      status: 'success',
      data: { sound },
    });
  } catch (error) {
    next(error);
  }
});

// Search sounds
router.get('/search', async (req, res, next) => {
  try {
    const query = req.query.q as string;
    const limit = parseInt(req.query.limit as string || '10');

    if (!query) {
      throw new AppError(400, 'Search query is required');
    }

    const sounds = await prisma.sound.findMany({
      where: {
        OR: [
          {
            name: {
              contains: query,
              mode: 'insensitive',
            },
          },
          {
            artistName: {
              contains: query,
              mode: 'insensitive',
            },
          },
        ],
      },
      take: limit,
      orderBy: {
        usageCount: 'desc',
      },
    });

    res.json({
      status: 'success',
      data: { sounds },
    });
  } catch (error) {
    next(error);
  }
});

// Get trending sounds
router.get('/trending', async (req, res, next) => {
  try {
    const limit = parseInt(req.query.limit as string || '10');

    const sounds = await prisma.sound.findMany({
      take: limit,
      orderBy: {
        usageCount: 'desc',
      },
      where: {
        usageCount: {
          gt: 0,
        },
      },
    });

    res.json({
      status: 'success',
      data: { sounds },
    });
  } catch (error) {
    next(error);
  }
});

// Get sound details and videos
router.get('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;
    const page = parseInt(req.query.page as string || '1');
    const limit = parseInt(req.query.limit as string || '10');
    const offset = (page - 1) * limit;

    const sound = await prisma.sound.findUnique({
      where: { id },
    });

    if (!sound) {
      throw new AppError(404, 'Sound not found');
    }

    const videos = await prisma.video.findMany({
      where: { soundId: id },
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
      where: { soundId: id },
    });

    res.json({
      status: 'success',
      data: {
        sound,
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

export { router as soundRouter }; 