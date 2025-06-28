'use client';

import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  Avatar, 
  Typography, 
  Button, 
  Space,
  Input,
  message,
  Empty,
  Spin,
  Badge,
  List,
  Tooltip
} from 'antd';
import { 
  UserOutlined, 
  SendOutlined,
  MessageOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  WifiOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../components/layout/AppLayout';
import { colors, cardStyle } from '../../styles/theme';
import { useRealtimeChat } from '../../hooks/useRealtimeChat';

dayjs.extend(relativeTime);

const { Text, Title } = Typography;
const { Search } = Input;

// Helper to access theme colors
const theme = colors.light;

interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  profilePicture?: string;
  isOnline?: boolean;
  lastSeen?: string;
}

interface Message {
  id: string;
  sender_id: string;
  receiver_id: string;
  content: string;
  created_at: string;
}

interface Conversation {
  id: string;
  otherUser: User;
  lastMessage?: Message;
  unreadCount: number;
}

export default function ChatPage() {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // State management
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messageInput, setMessageInput] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [sendingMessage, setSendingMessage] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [typingTimer, setTypingTimer] = useState<NodeJS.Timeout | null>(null);

  // Get token from localStorage
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    const storedToken = localStorage.getItem('token');
    setToken(storedToken);
  }, []);  // Real-time chat integration
  const { 
    isConnected, 
    messages: realtimeMessages,
    onlineUsers,
    typingUsers,
    sendMessage: realtimeSendMessage,
    setTyping,
    clearMessages
  } = useRealtimeChat(token || undefined);
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [realtimeMessages]);

  // Merge realtime messages with local messages
  useEffect(() => {
    setMessages(prev => {
      const allMessages = [...prev, ...realtimeMessages];
      const uniqueMessages = allMessages.filter((msg, index, self) => 
        index === self.findIndex(m => m.id === msg.id)
      );
      return uniqueMessages.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
    });
  }, [realtimeMessages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  // Load current user and conversations
  useEffect(() => {
    const loadUserData = async () => {
      try {
        if (!token) return;

        // Get current user from token
        const userResponse = await fetch('/api/auth/me', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
          if (userResponse.ok) {
          const userData = await userResponse.json();
          setCurrentUser(userData);
        } else {
          message.error('ไม่สามารถโหลดข้อมูลผู้ใช้ได้');
        }

        // Get conversations
        const conversationsResponse = await fetch('/api/chat/conversations', {
          headers: { 'Authorization': `Bearer ${token}` }
        });

        if (conversationsResponse.ok) {
          const conversationsData = await conversationsResponse.json();
          setConversations(conversationsData);
        } else {
          message.error('ไม่สามารถโหลดการสนทนาได้');
        }

      } catch (error) {
        console.error('Error loading user data:', error);
        message.error('เกิดข้อผิดพลาดในการโหลดข้อมูล');
      } finally {
        setLoading(false);
      }
    };

    loadUserData();
  }, [token]);
  // Load messages for selected conversation
  useEffect(() => {
    const loadMessages = async () => {
      if (!selectedConversation || !token) return;      try {
        const response = await fetch(`/api/chat/messages/${selectedConversation.otherUser.id}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
          const messagesData = await response.json();
          setMessages(messagesData);
          clearMessages(); // Clear socket messages to avoid duplicates
        } else {
          message.error('ไม่สามารถโหลดข้อความได้');
        }
      } catch (error) {
        console.error('Error loading messages:', error);
        message.error('เกิดข้อผิดพลาดในการโหลดข้อความ');
      }
    };

    loadMessages();
  }, [selectedConversation, token, clearMessages]);

  const handleSendMessage = async () => {
    if (!messageInput.trim() || !selectedConversation || !currentUser) return;

    setSendingMessage(true);
      try {
      // Send via Real-time API
      await realtimeSendMessage(selectedConversation.otherUser.id, messageInput.trim());
      setMessageInput('');
      
      // Stop typing indicator
      if (typingTimer) {
        clearTimeout(typingTimer);
        setTypingTimer(null);
      }
      setTyping(selectedConversation.otherUser.id, false);
      
    } catch (error) {
      console.error('Error sending message:', error);
      message.error('เกิดข้อผิดพลาดในการส่งข้อความ');
    } finally {
      setSendingMessage(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setMessageInput(e.target.value);
    
    // Typing indicator
    if (selectedConversation) {
      setTyping(selectedConversation.otherUser.id, true);
      
      // Clear previous timer
      if (typingTimer) {
        clearTimeout(typingTimer);
      }
      
      // Set new timer to stop typing
      const newTimer = setTimeout(() => {
        setTyping(selectedConversation.otherUser.id, false);
      }, 1000);
      
      setTypingTimer(newTimer);
    }
  };

  const handleSelectConversation = (conversation: Conversation) => {
    setSelectedConversation(conversation);
  };  const isUserOnline = (userId: string) => {
    const isOnline = onlineUsers.includes(userId);
    console.log('🟢 Online check:', { 
      userId, 
      userShortId: userId.substring(0, 8), 
      onlineUsers: onlineUsers.map(id => id.substring(0, 8)), 
      isOnline 
    });
    return isOnline;
  };

  const isUserTyping = (userId: string) => {
    return typingUsers.includes(userId);
  };

  // Filter conversations based on search
  const filteredConversations = conversations.filter(conversation =>
    conversation.otherUser.firstName.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conversation.otherUser.lastName.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conversation.otherUser.username.toLowerCase().includes(searchQuery.toLowerCase())
  );
  if (loading || !currentUser) {
    return (
      <AppLayout>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '400px',
          flexDirection: 'column',
          gap: '16px'
        }}>
          <Spin size="large" />
          <Text style={{ color: theme.textSecondary }}>
            {!currentUser ? 'กำลังโหลดข้อมูลผู้ใช้...' : 'กำลังโหลด...'}
          </Text>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div style={{ 
        height: 'calc(100vh - 140px)', 
        display: 'flex',
        gap: '16px'
      }}>
        {/* Sidebar - Conversations List */}
        <Card 
          style={{ 
            width: '350px', 
            height: '100%',
            ...cardStyle,
            display: 'flex',
            flexDirection: 'column'
          }}
          bodyStyle={{ 
            padding: 0, 
            height: '100%',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {/* Header */}
          <div style={{ 
            padding: '20px',
            borderBottom: `1px solid ${theme.border}`
          }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <MessageOutlined style={{ color: theme.primary }} />
                <Title level={4} style={{ margin: 0, color: theme.textPrimary }}>
                  แชท
                </Title>
                <Space>
                  {isConnected ? (
                    <WifiOutlined style={{ color: theme.success }} />
                  ) : (
                    <LoadingOutlined style={{ color: theme.warning }} />
                  )}
                </Space>
              </div>
              
              <Search
                placeholder="ค้นหาการสนทนา..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ width: '100%' }}
              />
            </Space>
          </div>

          {/* Conversations List */}
          <div style={{ 
            flex: 1, 
            overflow: 'auto',
            padding: '8px 0'
          }}>
            {filteredConversations.length > 0 ? (
              <List
                dataSource={filteredConversations}
                renderItem={(conversation) => (
                  <List.Item
                    style={{ 
                      padding: '12px 20px',
                      cursor: 'pointer',
                      backgroundColor: selectedConversation?.id === conversation.id
                        ? theme.bgHover
                        : 'transparent',
                      borderLeft: selectedConversation?.id === conversation.id
                        ? `3px solid ${theme.primary}`
                        : '3px solid transparent'
                    }}
                    onClick={() => handleSelectConversation(conversation)}
                  >
                    <List.Item.Meta
                      avatar={
                        <Badge 
                          dot 
                          status={isUserOnline(conversation.otherUser.id) ? 'success' : 'default'}
                          offset={[-8, 8]}
                        >
                          <Avatar 
                            src={conversation.otherUser.profilePicture}
                            icon={<UserOutlined />}
                            size={48}
                          />
                        </Badge>
                      }
                      title={
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Text strong style={{ color: theme.textPrimary }}>
                            {conversation.otherUser.firstName} {conversation.otherUser.lastName}
                          </Text>
                          {conversation.lastMessage && (
                            <Text 
                              style={{ 
                                fontSize: '12px',
                                color: theme.textSecondary,
                              }}
                            >
                              {dayjs(conversation.lastMessage.created_at).fromNow()}
                            </Text>
                          )}
                        </div>
                      }                      description={
                        <div>
                          <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            marginBottom: '4px'
                          }}>
                            <Text 
                              style={{ 
                                fontSize: '13px',
                                color: theme.textTertiary,
                              }}
                            >
                              {conversation.lastMessage?.content || 'ยังไม่มีข้อความ'}
                            </Text>
                            {conversation.unreadCount > 0 && (
                              <Badge 
                                count={conversation.unreadCount} 
                                style={{ marginLeft: '8px' }}
                              />
                            )}
                          </div>                          <Text 
                            style={{ 
                              fontSize: '11px',
                              color: isUserOnline(conversation.otherUser.id) ? theme.success : theme.textSecondary,
                              fontWeight: isUserOnline(conversation.otherUser.id) ? 'bold' : 'normal'
                            }}
                          >
                            {isUserOnline(conversation.otherUser.id) 
                              ? 'ออนไลน์' 
                              : 'ออฟไลน์'
                            }
                          </Text>
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            ) : (
              <div style={{ 
                padding: '40px 20px', 
                textAlign: 'center' 
              }}>
                <Empty 
                  description={
                    <Text style={{ color: theme.textPrimary }}>
                      {searchQuery ? 'ไม่พบการสนทนาที่ค้นหา' : 'ยังไม่มีการสนทนา'}
                    </Text>
                  }
                />
                <Text style={{ color: theme.textSecondary, fontSize: '12px' }}>
                  เริ่มต้นการสนทนาใหม่โดยค้นหาผู้ใช้
                </Text>
              </div>
            )}
          </div>
        </Card>

        {/* Main Chat Area */}
        <Card 
          style={{ 
            flex: 1, 
            height: '100%',
            ...cardStyle,
            display: 'flex',
            flexDirection: 'column'
          }}
          bodyStyle={{ 
            padding: 0, 
            height: '100%',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {selectedConversation ? (
            <>
              {/* Chat Header */}
              <div style={{ 
                padding: '16px 20px',
                borderBottom: `1px solid ${theme.border}`,
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <Badge 
                  dot 
                  status={isUserOnline(selectedConversation.otherUser.id) ? 'success' : 'default'}
                  offset={[-8, 8]}
                >
                  <Avatar 
                    src={selectedConversation.otherUser.profilePicture}
                    icon={<UserOutlined />}
                    size={40}
                  />
                </Badge>
                <div style={{ flex: 1 }}>
                  <Title level={5} style={{ margin: 0, color: theme.textPrimary }}>
                    {selectedConversation.otherUser.firstName} {selectedConversation.otherUser.lastName}
                  </Title>
                  <Text style={{ color: theme.textSecondary, fontSize: '12px' }}>
                    {isUserOnline(selectedConversation.otherUser.id) 
                      ? 'ออนไลน์' 
                      : selectedConversation.otherUser.lastSeen 
                        ? `ออนไลน์ล่าสุด ${dayjs(selectedConversation.otherUser.lastSeen).fromNow()}`
                        : 'ออฟไลน์'
                    }
                  </Text>
                </div>
                <Space>
                  {isConnected ? (
                    <Tooltip title="เชื่อมต่อแล้ว">
                      <CheckCircleOutlined style={{ color: theme.success }} />
                    </Tooltip>
                  ) : (
                    <Tooltip title="กำลังเชื่อมต่อ...">
                      <LoadingOutlined style={{ color: theme.warning }} />
                    </Tooltip>
                  )}
                </Space>
              </div>

              {/* Messages Area */}
              <div style={{ 
                flex: 1, 
                overflow: 'auto',
                padding: '16px 20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px'
              }}>                {messages.map((message) => {
                  const isOwn = message.sender_id === currentUser?.id;
                  
                  return (
                    <div
                      key={message.id}
                      style={{
                        display: 'flex',
                        justifyContent: isOwn ? 'flex-end' : 'flex-start',
                        alignItems: 'flex-end',
                        gap: '8px'
                      }}
                    >                      {!isOwn && (
                        <Avatar 
                          src={selectedConversation.otherUser.profilePicture}
                          icon={<UserOutlined />}
                          size={32}
                        />
                      )}
                      
                      <div style={{ maxWidth: '70%' }}>
                        <div
                          style={{
                            padding: '8px 12px',
                            borderRadius: '12px',
                            backgroundColor: isOwn ? theme.primary : theme.bgSecondary,
                            color: isOwn ? 'white' : theme.textPrimary,
                          }}
                        >
                          <Text style={{ 
                            color: isOwn ? 'white' : theme.textPrimary
                          }}>
                            {message.content}
                          </Text>
                        </div>
                        <Text 
                          style={{ 
                            fontSize: '11px',
                            color: isOwn ? theme.textTertiary : theme.textTertiary,
                            display: 'block',
                            textAlign: isOwn ? 'right' : 'left',
                            marginTop: '4px'
                          }}
                        >
                          {dayjs(message.created_at).format('HH:mm')}
                        </Text>
                      </div>
                      
                      {isOwn && (
                        <Avatar 
                          src={currentUser?.profilePicture}
                          icon={<UserOutlined />}
                          size={32}
                        />
                      )}
                    </div>
                  );
                })}

                {/* Typing Indicator */}
                {isUserTyping(selectedConversation.otherUser.id) && (
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px',
                    padding: '8px 12px'
                  }}>
                    <Avatar 
                      src={selectedConversation.otherUser.profilePicture}
                      icon={<UserOutlined />}
                      size={32}
                    />
                    <div
                      style={{
                        padding: '8px 12px',
                        borderRadius: '12px',
                        backgroundColor: theme.bgSecondary,
                        color: theme.textSecondary
                      }}
                    >
                      <Text style={{ fontStyle: 'italic' }}>กำลังพิมพ์...</Text>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Message Input */}
              <div style={{ 
                padding: '16px 20px',
                borderTop: `1px solid ${theme.border}`
              }}>
                <Space.Compact style={{ width: '100%' }}>
                  <Input
                    value={messageInput}
                    onChange={handleInputChange}
                    placeholder="พิมพ์ข้อความ..."
                    onPressEnter={handleSendMessage}
                    disabled={sendingMessage || !isConnected}
                    style={{ flex: 1 }}
                  />
                  <Button
                    type="primary"
                    icon={<SendOutlined />}
                    onClick={handleSendMessage}
                    loading={sendingMessage}
                    disabled={!messageInput.trim() || !isConnected}
                  >
                    ส่ง
                  </Button>
                </Space.Compact>
              </div>
            </>
          ) : (
            /* No Conversation Selected */
            <div style={{ 
              flex: 1, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              flexDirection: 'column',
              gap: '16px',
              color: theme.textSecondary
            }}>
              <MessageOutlined style={{ fontSize: '64px' }} />
              <Title level={4} style={{ color: theme.textSecondary }}>
                เลือกการสนทนาเพื่อเริ่มต้นแชท
              </Title>
              <Text style={{ color: theme.textTertiary }}>
                เลือกผู้ติดต่อจากรายการทางซ้ายเพื่อเริ่มการสนทนา
              </Text>
            </div>
          )}
        </Card>
      </div>
    </AppLayout>
  );
}
