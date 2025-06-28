import { NextRequest } from 'next/server';
import { PrismaClient } from '@prisma/client';
import * as jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

// Store active connections
const connections = new Map<string, ReadableStreamDefaultController>();

export async function GET(request: NextRequest) {
  try {
    // Get token from query params
    const { searchParams } = new URL(request.url);
    const token = searchParams.get('token');
    
    if (!token) {
      return new Response('Unauthorized', { status: 401 });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    const userId = decoded.userId;

    // Create Server-Sent Events stream
    const stream = new ReadableStream({
      start(controller) {        // Store connection
        connections.set(userId, controller);
        
        // Send initial connection message
        controller.enqueue(`data: ${JSON.stringify({ 
          type: 'connected', 
          userId: userId,
          timestamp: new Date().toISOString()
        })}\n\n`);

        // Send current online users list to the new connection
        const currentOnlineUsers = Array.from(connections.keys());
        controller.enqueue(`data: ${JSON.stringify({ 
          type: 'online-users-list', 
          onlineUsers: currentOnlineUsers,
          timestamp: new Date().toISOString()
        })}\n\n`);

        // Broadcast user online status to all other users (exclude self)
        broadcastToOthers(userId, {
          type: 'user-status',
          userId: userId,
          isOnline: true,
          timestamp: new Date().toISOString()
        });

        // Keep connection alive with heartbeat
        const heartbeat = setInterval(() => {
          try {
            controller.enqueue(`data: ${JSON.stringify({ 
              type: 'heartbeat', 
              timestamp: new Date().toISOString() 
            })}\n\n`);
          } catch (error) {
            clearInterval(heartbeat);
            connections.delete(userId);
            // Broadcast user offline status
            broadcastToAll({
              type: 'user-status',
              userId: userId,
              isOnline: false,
              timestamp: new Date().toISOString()
            });
          }
        }, 30000); // 30 seconds        // Cleanup on close
        request.signal.addEventListener('abort', () => {
          clearInterval(heartbeat);
          connections.delete(userId);
          // Broadcast user offline status to others
          broadcastToOthers(userId, {
            type: 'user-status',
            userId: userId,
            isOnline: false,
            timestamp: new Date().toISOString()
          });
          try {
            controller.close();
          } catch (error) {
            // Connection already closed
          }
        });
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
      }
    });

  } catch (error) {
    console.error('SSE connection error:', error);
    return new Response('Invalid token', { status: 401 });
  }
}

// Helper function to send message to specific user
export function sendToUser(userId: string, data: any) {
  const controller = connections.get(userId);
  if (controller) {
    try {
      controller.enqueue(`data: ${JSON.stringify(data)}\n\n`);
    } catch (error) {
      console.error('Error sending to user:', userId, error);
      connections.delete(userId);
    }
  }
}

// Helper function to broadcast to all connected users
export function broadcastToAll(data: any) {
  for (const [userId, controller] of connections.entries()) {
    try {
      controller.enqueue(`data: ${JSON.stringify(data)}\n\n`);
    } catch (error) {
      console.error('Error broadcasting to user:', userId, error);
      connections.delete(userId);
    }
  }
}

// Helper function to broadcast to all users except the specified one
export function broadcastToOthers(excludeUserId: string, data: any) {
  for (const [userId, controller] of connections.entries()) {
    if (userId !== excludeUserId) {
      try {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`);
      } catch (error) {
        console.error('Error broadcasting to user:', userId, error);
        connections.delete(userId);
      }
    }
  }
}
