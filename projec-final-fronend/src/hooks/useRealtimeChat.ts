import { useEffect, useState, useCallback, useRef } from 'react';

interface Message {
  id: string;
  sender_id: string;
  receiver_id: string;
  content: string;
  created_at: string;
  sender?: {
    id: string;
    username: string;
    firstName: string;
    lastName: string;
    profilePicture?: string;
  };
}

interface UseRealtimeChatReturn {
  isConnected: boolean;
  messages: Message[];
  onlineUsers: string[];
  typingUsers: string[];
  sendMessage: (receiverId: string, content: string) => Promise<void>;
  setTyping: (receiverId: string, isTyping: boolean) => void;
  clearMessages: () => void;
}

export const useRealtimeChat = (token?: string): UseRealtimeChatReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [onlineUsers, setOnlineUsers] = useState<string[]>([]);
  const [typingUsers, setTypingUsers] = useState<string[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);
  const typingTimeoutRef = useRef<{ [key: string]: NodeJS.Timeout }>({});
  useEffect(() => {
    if (!token) return;

    console.log('🔌 Connecting to SSE stream...');
    
    // Create EventSource connection for Server-Sent Events
    const eventSource = new EventSource(`/api/chat/stream?token=${encodeURIComponent(token)}`);
    eventSourceRef.current = eventSource;    eventSource.onopen = () => {
      console.log('✅ SSE connection opened');
      setIsConnected(true);
      
      // Send initial online status request to get current online users
      setTimeout(() => {
        if (eventSourceRef.current && eventSourceRef.current.readyState === EventSource.OPEN) {
          console.log('📡 Requesting current online users...');
        }
      }, 1000);
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📨 SSE message received:', data);

        switch (data.type) {          case 'connected':
            console.log('🔗 Connected to chat stream');
            setIsConnected(true);
            break;

          case 'online-users-list':
            console.log('📋 Received online users list:', data.onlineUsers);
            setOnlineUsers(data.onlineUsers);
            break;

          case 'new-message':
            console.log('💬 New message received:', data.message);
            setMessages(prev => [...prev, data.message]);
            break;

          case 'message-sent':
            // Update local state to show sent message
            console.log('📤 Message sent confirmation:', data.message);
            setMessages(prev => {
              const exists = prev.find(m => m.id === data.message.id);
              if (!exists) {
                return [...prev, data.message];
              }
              return prev;
            });
            break;

          case 'user-typing':
            console.log('⌨️ User typing status:', data);
            setTypingUsers(prev => {
              if (data.isTyping) {
                return prev.includes(data.userId) ? prev : [...prev, data.userId];
              } else {
                return prev.filter(id => id !== data.userId);
              }
            });
            break;          case 'user-status':
            console.log('👤 User status update:', data);
            setOnlineUsers(prev => {
              const updatedUsers = data.isOnline 
                ? (prev.includes(data.userId) ? prev : [...prev, data.userId])
                : prev.filter(id => id !== data.userId);
              console.log('👥 Updated online users:', updatedUsers);
              return updatedUsers;
            });
            break;

          case 'heartbeat':
            // Keep connection alive
            console.log('💓 Heartbeat received');
            break;

          default:
            console.log('❓ Unknown SSE message type:', data.type);
        }
      } catch (error) {
        console.error('❌ Error parsing SSE message:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('❌ SSE connection error:', error);
      setIsConnected(false);
    };

    // Cleanup on unmount
    return () => {
      console.log('🔌 Closing SSE connection');
      eventSource.close();
      eventSourceRef.current = null;
      setIsConnected(false);
      
      // Clear typing timeouts
      Object.values(typingTimeoutRef.current).forEach(clearTimeout);
      typingTimeoutRef.current = {};
    };
  }, [token]);

  // Send message function
  const sendMessage = useCallback(async (receiverId: string, content: string) => {
    if (!token) return;

    try {
      const response = await fetch('/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ receiverId, content })
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      console.log('Message sent successfully');
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }, [token]);

  // Typing indicator (simplified - could be enhanced with API calls)
  const setTyping = useCallback((receiverId: string, isTyping: boolean) => {
    if (!token) return;

    // Clear existing timeout for this user
    if (typingTimeoutRef.current[receiverId]) {
      clearTimeout(typingTimeoutRef.current[receiverId]);
    }

    if (isTyping) {
      // Set timeout to automatically stop typing after 3 seconds
      typingTimeoutRef.current[receiverId] = setTimeout(() => {
        // Could send API call to stop typing indicator
      }, 3000);
    }

    // For now, just log typing status (could be enhanced with API calls)
    console.log(`User typing: ${receiverId}, isTyping: ${isTyping}`);
  }, [token]);

  // Clear messages function
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    isConnected,
    messages,
    onlineUsers,
    typingUsers,
    sendMessage,
    setTyping,
    clearMessages
  };
};
