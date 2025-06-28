import { NextApiRequest, NextApiResponse } from 'next';
import { Server } from 'socket.io';
import { PrismaClient } from '@prisma/client';
import * as jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export default function handler(req: NextApiRequest, res: any) {
  if (res.socket?.server?.io) {
    console.log('Socket.IO server already running');
    res.end();
    return;
  }

  console.log('Starting Socket.IO server...');
  const io = new Server(res.socket?.server, {
    path: '/api/socket',
    addTrailingSlash: false,
    cors: {
      origin: "*",
      methods: ["GET", "POST"]
    }
  });

  res.socket.server.io = io;

  io.on('connection', (socket: any) => {
    console.log('User connected:', socket.id);

    // Authentication
    socket.on('authenticate', async (token: string) => {
      try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
        socket.userId = decoded.userId;
        socket.join(`user-${decoded.userId}`);
        console.log(`User ${decoded.userId} authenticated`);
        socket.emit('authenticated', { success: true });
      } catch (error) {
        console.error('Authentication failed:', error);
        socket.emit('auth-error', { message: 'Invalid token' });
      }
    });

    // Join conversation room
    socket.on('join-conversation', (conversationId: string) => {
      socket.join(`conversation-${conversationId}`);
      console.log(`User joined conversation: ${conversationId}`);
    });

    // Send message
    socket.on('send-message', async (data: any) => {
      const { receiverId, content } = data;
      
      if (!socket.userId) {
        socket.emit('error', { message: 'Not authenticated' });
        return;
      }

      try {        // Save message to database
        const savedMessage = await (prisma as any).message.create({
          data: {
            senderId: socket.userId,
            receiverId: receiverId,
            content: content
          }
        });

        // Send to both sender and receiver
        const messageData = {
          id: savedMessage.id,
          sender_id: savedMessage.senderId,
          receiver_id: savedMessage.receiverId,
          content: savedMessage.content,
          created_at: savedMessage.createdAt
        };

        // Send to sender
        socket.emit('message-sent', messageData);
        
        // Send to receiver
        socket.to(`user-${receiverId}`).emit('new-message', messageData);

        console.log('Message sent successfully');
      } catch (error) {
        console.error('Error saving message:', error);
        socket.emit('error', { message: 'Failed to send message' });
      }
    });

    // Typing indicator
    socket.on('typing', (data: any) => {
      socket.to(`user-${data.receiverId}`).emit('user-typing', {
        userId: socket.userId,
        isTyping: true
      });
    });

    socket.on('stop-typing', (data: any) => {
      socket.to(`user-${data.receiverId}`).emit('user-typing', {
        userId: socket.userId,
        isTyping: false
      });
    });

    // Online status
    socket.on('user-online', () => {
      if (socket.userId) {
        socket.broadcast.emit('user-status', {
          userId: socket.userId,
          isOnline: true
        });
      }
    });

    socket.on('disconnect', () => {
      console.log('User disconnected:', socket.id);
      if (socket.userId) {
        socket.broadcast.emit('user-status', {
          userId: socket.userId,
          isOnline: false
        });
      }
    });
  });

  res.end();
}
