import { WebSocket, WebSocketServer } from 'ws';
import { logger } from './utils/logger';
import { redis } from './index';

interface WebSocketWithId extends WebSocket {
  id?: string;
  userId?: string;
}

interface WebSocketMessage {
  type: string;
  payload: any;
}

export const setupWebSocketHandlers = (wss: WebSocketServer) => {
  // Store active connections
  const connections = new Map<string, WebSocketWithId>();

  wss.on('connection', async (ws: WebSocketWithId) => {
    ws.id = Math.random().toString(36).substring(2, 15);
    connections.set(ws.id, ws);

    logger.info(`New WebSocket connection: ${ws.id}`);

    // Handle incoming messages
    ws.on('message', async (data: string) => {
      try {
        const message: WebSocketMessage = JSON.parse(data);

        switch (message.type) {
          case 'auth':
            ws.userId = message.payload.userId;
            // Subscribe to user's notifications channel
            await redis.subscribe(`notifications:${ws.userId}`, (message) => {
              ws.send(JSON.stringify({
                type: 'notification',
                payload: JSON.parse(message)
              }));
            });
            break;

          case 'videoEngagement':
            // Handle video engagement events (likes, comments, shares)
            if (ws.userId) {
              await redis.publish('videoEngagements', JSON.stringify({
                userId: ws.userId,
                ...message.payload
              }));
            }
            break;

          default:
            logger.warn(`Unknown message type: ${message.type}`);
        }
      } catch (error) {
        logger.error('Error processing WebSocket message:', error);
      }
    });

    // Handle client disconnect
    ws.on('close', async () => {
      if (ws.id) {
        connections.delete(ws.id);
      }
      if (ws.userId) {
        await redis.unsubscribe(`notifications:${ws.userId}`);
      }
      logger.info(`Client disconnected: ${ws.id}`);
    });

    // Send initial connection success message
    ws.send(JSON.stringify({
      type: 'connected',
      payload: { id: ws.id }
    }));
  });

  // Broadcast to all connected clients
  const broadcast = (message: WebSocketMessage) => {
    connections.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(message));
      }
    });
  };

  return {
    connections,
    broadcast
  };
}; 