import { useEffect, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

interface Message {
  id: string;
  sender_id: string;
  receiver_id: string;
  content: string;
  created_at: string;
}

interface UseSocketReturn {
  socket: Socket | null;
  isConnected: boolean;
  messages: Message[];
  onlineUsers: string[];
  typingUsers: string[];
  sendMessage: (receiverId: string, content: string) => void;
  setTyping: (receiverId: string, isTyping: boolean) => void;
  clearMessages: () => void;
}

export const useSocket = (token?: string): UseSocketReturn => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [onlineUsers, setOnlineUsers] = useState<string[]>([]);
  const [typingUsers, setTypingUsers] = useState<string[]>([]);

  useEffect(() => {
    if (!token) return;

    // เชื่อมต่อ Socket.IO
    const socketConnection = io({
      path: '/api/socket',
      addTrailingSlash: false,
    });

    setSocket(socketConnection);

    // เชื่อมต่อสำเร็จ
    socketConnection.on('connect', () => {
      console.log('Connected to Socket.IO server');
      setIsConnected(true);
      
      // ส่ง token เพื่อ authenticate
      socketConnection.emit('authenticate', token);
    });

    // Authentication สำเร็จ
    socketConnection.on('authenticated', (data) => {
      console.log('Authentication successful:', data);
      socketConnection.emit('user-online');
    });

    // Authentication ผิดพลาด
    socketConnection.on('auth-error', (error) => {
      console.error('Authentication failed:', error);
    });

    // รับข้อความใหม่
    socketConnection.on('new-message', (message: Message) => {
      console.log('New message received:', message);
      setMessages(prev => [...prev, message]);
    });

    // ข้อความส่งสำเร็จ
    socketConnection.on('message-sent', (message: Message) => {
      console.log('Message sent successfully:', message);
      setMessages(prev => [...prev, message]);
    });

    // สถานะการพิมพ์
    socketConnection.on('user-typing', (data: { userId: string; isTyping: boolean }) => {
      setTypingUsers(prev => {
        if (data.isTyping) {
          return prev.includes(data.userId) ? prev : [...prev, data.userId];
        } else {
          return prev.filter(id => id !== data.userId);
        }
      });
    });

    // สถานะออนไลน์
    socketConnection.on('user-status', (data: { userId: string; isOnline: boolean }) => {
      setOnlineUsers(prev => {
        if (data.isOnline) {
          return prev.includes(data.userId) ? prev : [...prev, data.userId];
        } else {
          return prev.filter(id => id !== data.userId);
        }
      });
    });

    // ข้อผิดพลาด
    socketConnection.on('error', (error) => {
      console.error('Socket error:', error);
    });

    // การตัดการเชื่อมต่อ
    socketConnection.on('disconnect', () => {
      console.log('Disconnected from Socket.IO server');
      setIsConnected(false);
    });

    // Cleanup เมื่อ component unmount
    return () => {
      socketConnection.close();
    };
  }, [token]);

  // ส่งข้อความ
  const sendMessage = useCallback((receiverId: string, content: string) => {
    if (socket && isConnected) {
      socket.emit('send-message', { receiverId, content });
    }
  }, [socket, isConnected]);

  // ตั้งค่าสถานะการพิมพ์
  const setTyping = useCallback((receiverId: string, isTyping: boolean) => {
    if (socket && isConnected) {
      if (isTyping) {
        socket.emit('typing', { receiverId });
      } else {
        socket.emit('stop-typing', { receiverId });
      }
    }
  }, [socket, isConnected]);

  // ลบข้อความทั้งหมด
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    socket,
    isConnected,
    messages,
    onlineUsers,
    typingUsers,
    sendMessage,
    setTyping,
    clearMessages
  };
};
