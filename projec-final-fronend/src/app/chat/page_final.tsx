'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  Tooltip
} from 'antd';
import { 
  UserOutlined, 
  SendOutlined,
  SearchOutlined,
  TeamOutlined,
  MessageOutlined,
  PhoneOutlined,
  VideoCameraOutlined,
  MoreOutlined,
  SmileOutlined,
  LoadingOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../components/layout/AppLayout';
import { colors, cardStyle, textStyle } from '../../styles/theme';

dayjs.extend(relativeTime);

const { Text, Title } = Typography;
const { Search } = Input;

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
  content: string;
  sender_id: string;
  receiver_id: string;
  read: boolean;
  created_at: string;
  sender?: User;
  isTemp?: boolean;
}

interface Conversation {
  user: User;
  lastMessage?: string;
  lastMessageTime?: string;
  unreadCount: number;
}

const ChatPage: React.FC = () => {
  const router = useRouter();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesCache = useRef<{ [userId: string]: Message[] }>({});
  
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isActive, setIsActive] = useState(true);
  const [lastMessageTime, setLastMessageTime] = useState(0);

  // Track user activity for smart polling
  useEffect(() => {
    const handleActivity = () => setIsActive(true);
    const handleInactivity = () => setIsActive(false);

    window.addEventListener('focus', handleActivity);
    window.addEventListener('blur', handleInactivity);
    window.addEventListener('mousemove', handleActivity);
    window.addEventListener('keypress', handleActivity);
    
    return () => {
      window.removeEventListener('focus', handleActivity);
      window.removeEventListener('blur', handleInactivity);
      window.removeEventListener('mousemove', handleActivity);
      window.removeEventListener('keypress', handleActivity);
    };
  }, []);

  const fetchCurrentUser = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const userData = await response.json();
        setCurrentUser(userData);
      }
    } catch (error) {
      console.error('Error fetching user:', error);
    }
  }, [router]);

  const fetchConversations = useCallback(async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      const response = await fetch('/api/chat/conversations', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setConversations(data);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchMessages = useCallback(async (userId: string, forceRefresh = false) => {
    try {
      // ใช้ cache ถ้าไม่ได้บังคับ refresh และมีข้อมูลใน cache
      if (!forceRefresh && messagesCache.current[userId]) {
        setMessages(messagesCache.current[userId]);
        return;
      }

      setMessagesLoading(true);
      const token = localStorage.getItem('token');

      const response = await fetch(`/api/chat/messages/${userId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(data);
        // อัพเดท cache
        messagesCache.current[userId] = data;
        setLastMessageTime(Date.now());
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    } finally {
      setMessagesLoading(false);
    }
  }, []);

  // Load initial data
  useEffect(() => {
    fetchCurrentUser();
    fetchConversations();
  }, [fetchCurrentUser, fetchConversations]);

  useEffect(() => {
    if (selectedConversation) {
      fetchMessages(selectedConversation.user.id, false);
    }
  }, [selectedConversation, fetchMessages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Smart polling: ทำงานเฉพาะเมื่อ user active และมีการสนทนา
  useEffect(() => {
    if (!selectedConversation || !isActive) return;
    
    const interval = setInterval(() => {
      // ใช้ forceRefresh เฉพาะครั้งแรก หรือถ้าผ่านไป 30 วินาทีแล้ว
      const shouldForceRefresh = Date.now() - lastMessageTime > 30000;
      fetchMessages(selectedConversation.user.id, shouldForceRefresh);
    }, 3000); // เช็คทุก 3 วินาที

    return () => clearInterval(interval);
  }, [selectedConversation, isActive, lastMessageTime, fetchMessages]);

  // Update conversations less frequently
  useEffect(() => {
    if (!isActive) return;
    
    const interval = setInterval(() => {
      fetchConversations();
    }, 10000); // ทุก 10 วินาที

    return () => clearInterval(interval);
  }, [isActive, fetchConversations]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !selectedConversation || sending) return;

    const messageContent = newMessage.trim();
    const tempMessage: Message = {
      id: `temp-${Date.now()}`,
      content: messageContent,
      sender_id: currentUser?.id || '',
      receiver_id: selectedConversation.user.id,
      created_at: new Date().toISOString(),
      read: false,
      isTemp: true
    };

    // Add message to UI immediately (optimistic update)
    setMessages(prev => [...prev, tempMessage]);
    setNewMessage('');

    try {
      setSending(true);

      const token = localStorage.getItem('token');
      const response = await fetch('/api/chat/messages', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          receiverId: selectedConversation.user.id,
          content: messageContent,
        }),
      });

      if (response.ok) {
        const newMsg = await response.json();
        
        // Replace temp message with real one
        setMessages(prev => prev.map(msg => 
          msg.id === tempMessage.id ? { ...newMsg, isTemp: false } : msg
        ));
        
        // Clear cache และ refresh
        messagesCache.current[selectedConversation.user.id] = [];
        fetchMessages(selectedConversation.user.id, true);
        fetchConversations();
      } else {
        // Remove temp message on failure
        setMessages(prev => prev.filter(msg => msg.id !== tempMessage.id));
        setNewMessage(messageContent);
        message.error('ไม่สามารถส่งข้อความได้');
      }
    } catch (error) {
      // Remove temp message on error
      setMessages(prev => prev.filter(msg => msg.id !== tempMessage.id));
      setNewMessage(messageContent);
      console.error('Error sending message:', error);
      message.error('เกิดข้อผิดพลาดในการส่งข้อความ');
    } finally {
      setSending(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const filteredConversations = conversations.filter(conv =>
    conv.user.firstName.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conv.user.lastName.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conv.user.username.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const ConversationItem: React.FC<{ conversation: Conversation }> = ({ conversation }) => {
    const isSelected = selectedConversation?.user.id === conversation.user.id;
    
    return (
      <div
        style={{
          padding: '16px',
          cursor: 'pointer',
          backgroundColor: isSelected ? `${colors.light.primary}08` : 'transparent',
          borderLeft: isSelected ? `3px solid ${colors.light.primary}` : '3px solid transparent',
          transition: 'all 0.2s ease',
        }}
        onClick={() => {
          setSelectedConversation(conversation);
          // Clear cache เมื่อเปลี่ยนการสนทนา
          messagesCache.current = {};
        }}
        onMouseEnter={(e) => {
          if (!isSelected) {
            e.currentTarget.style.backgroundColor = colors.light.bgHover;
          }
        }}
        onMouseLeave={(e) => {
          if (!isSelected) {
            e.currentTarget.style.backgroundColor = 'transparent';
          }
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ position: 'relative', marginRight: '12px' }}>
            <Avatar
              size={48}
              src={conversation.user.profilePicture}
              icon={<UserOutlined />}
            />
            {conversation.user.isOnline && (
              <div style={{
                position: 'absolute',
                bottom: '2px',
                right: '2px',
                width: '12px',
                height: '12px',
                backgroundColor: colors.light.success,
                borderRadius: '50%',
                border: `2px solid ${colors.light.bgContainer}`,
              }} />
            )}
          </div>
          
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Text 
                strong 
                style={{ 
                  ...textStyle.body,
                  fontWeight: 600,
                  color: colors.light.textPrimary
                }}
              >
                {conversation.user.firstName} {conversation.user.lastName}
              </Text>
              {conversation.unreadCount > 0 && (
                <Badge 
                  count={conversation.unreadCount} 
                  size="small"
                  style={{ backgroundColor: colors.light.primary }}
                />
              )}
            </div>
            
            <Text 
              style={{ 
                ...textStyle.caption,
                color: colors.light.textSecondary,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                display: 'block'
              }}
            >
              @{conversation.user.username}
            </Text>
            
            {conversation.lastMessage && (
              <Text 
                style={{ 
                  ...textStyle.caption,
                  color: colors.light.textTertiary,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  display: 'block',
                  marginTop: '2px'
                }}
              >
                {typeof conversation.lastMessage === 'string' 
                  ? conversation.lastMessage 
                  : 'ข้อความ...'}
              </Text>
            )}
          </div>
        </div>
      </div>
    );
  };

  const MessageItem: React.FC<{ message: Message }> = ({ message }) => {
    const senderId = String(message.sender_id).trim();
    const currentUserId = String(currentUser?.id || '').trim();
    const isOwn = senderId === currentUserId;
    
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: isOwn ? 'flex-end' : 'flex-start',
        marginBottom: '16px',
        width: '100%',
        padding: '4px 8px'
      }}>
        {!isOwn && (
          <Avatar
            size={32}
            src={selectedConversation?.user.profilePicture}
            icon={<UserOutlined />}
            style={{ marginRight: '8px', marginTop: '4px', flexShrink: 0 }}
          />
        )}
        
        <div style={{ 
          maxWidth: '70%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: isOwn ? 'flex-end' : 'flex-start'
        }}>
          <Text style={{ 
            fontSize: '11px',
            color: colors.light.textTertiary,
            marginBottom: '2px',
            textAlign: isOwn ? 'right' : 'left'
          }}>
            {isOwn ? 'คุณ' : selectedConversation?.user.firstName}
          </Text>
          
          <div style={{
            padding: '12px 16px',
            borderRadius: isOwn ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
            backgroundColor: isOwn ? colors.light.primary : '#f8f9fa',
            color: isOwn ? '#ffffff' : colors.light.textPrimary,
            maxWidth: '100%',
            wordWrap: 'break-word',
            boxShadow: `0 1px 3px ${colors.light.shadowLight}`,
            position: 'relative',
            border: isOwn ? 'none' : `1px solid ${colors.light.border}`,
            opacity: message.isTemp ? 0.7 : 1
          }}>
            <Text style={{ 
              color: isOwn ? '#ffffff' : colors.light.textPrimary,
              fontSize: '14px',
              lineHeight: 1.4
            }}>
              {message.content}
            </Text>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', marginTop: '4px' }}>
            <Text style={{ 
              ...textStyle.caption,
              color: colors.light.textTertiary,
              fontSize: '11px'
            }}>
              {dayjs(message.created_at).format('HH:mm')}
            </Text>
            
            {isOwn && !message.isTemp && (
              <CheckCircleOutlined 
                style={{ 
                  color: colors.light.success, 
                  fontSize: '10px', 
                  marginLeft: '4px' 
                }} 
              />
            )}
          </div>
        </div>
        
        {isOwn && (
          <Avatar
            size={32}
            src={currentUser?.profilePicture}
            icon={<UserOutlined />}
            style={{ marginLeft: '8px', marginTop: '4px', flexShrink: 0 }}
          />
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <AppLayout>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          minHeight: '400px' 
        }}>
          <Spin size="large" />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <style>{`
        @keyframes pulse {
          0% { opacity: 0.5; }
          50% { opacity: 1; }
          100% { opacity: 0.5; }
        }
        .connection-indicator {
          animation: pulse 2s infinite;
        }
      `}</style>
      
      <div style={{ height: 'calc(100vh - 140px)', display: 'flex' }}>
        {/* Conversations List */}
        <Card
          style={{
            ...cardStyle,
            width: '380px',
            height: '100%',
            marginRight: '16px',
            display: 'flex',
            flexDirection: 'column'
          }}
          bodyStyle={{ padding: 0, height: '100%', display: 'flex', flexDirection: 'column' }}
        >
          <div style={{ 
            padding: '20px', 
            borderBottom: `1px solid ${colors.light.border}`,
            backgroundColor: colors.light.bgContainer
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <Title level={4} style={{ margin: 0, color: colors.light.textPrimary }}>
                แชท
              </Title>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Tooltip title={isActive ? "กำลังใช้งาน" : "ไม่ได้ใช้งาน"}>
                  <div 
                    className={!isActive ? "connection-indicator" : ""}
                    style={{
                      width: '8px',
                      height: '8px',
                      backgroundColor: isActive ? colors.light.success : colors.light.warning,
                      borderRadius: '50%'
                    }} 
                  />
                </Tooltip>
                <Button
                  type="text"
                  icon={<TeamOutlined style={{ color: colors.light.textSecondary, fontSize: '16px' }} />}
                  style={{ 
                    borderRadius: '8px',
                    border: 'none',
                    backgroundColor: 'transparent'
                  }}
                />
              </div>
            </div>
            
            <Search
              placeholder="ค้นหาแชท..."
              prefix={<SearchOutlined style={{ color: colors.light.textTertiary }} />}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ borderRadius: '8px' }}
            />
          </div>

          <div style={{ flex: 1, overflow: 'auto' }}>
            {filteredConversations.length === 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="ไม่มีการสนทนา"
                style={{ margin: '40px 0' }}
              />
            ) : (
              filteredConversations.map((conversation) => (
                <ConversationItem key={conversation.user.id} conversation={conversation} />
              ))
            )}
          </div>
        </Card>

        {/* Chat Area */}
        <Card
          style={{
            ...cardStyle,
            flex: 1,
            height: '100%',
            display: 'flex',
            flexDirection: 'column'
          }}
          bodyStyle={{ padding: 0, height: '100%', display: 'flex', flexDirection: 'column' }}
        >
          {selectedConversation ? (
            <>
              <div style={{ 
                padding: '20px',
                borderBottom: `1px solid ${colors.light.border}`,
                backgroundColor: colors.light.bgContainer,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <div style={{ position: 'relative', marginRight: '12px' }}>
                    <Avatar
                      size={40}
                      src={selectedConversation.user.profilePicture}
                      icon={<UserOutlined />}
                    />
                    {selectedConversation.user.isOnline && (
                      <div style={{
                        position: 'absolute',
                        bottom: '2px',
                        right: '2px',
                        width: '10px',
                        height: '10px',
                        backgroundColor: colors.light.success,
                        borderRadius: '50%',
                        border: `2px solid ${colors.light.bgContainer}`,
                      }} />
                    )}
                  </div>
                  
                  <div>
                    <Text 
                      strong 
                      style={{ 
                        ...textStyle.body,
                        color: colors.light.textPrimary,
                        fontSize: '16px',
                        display: 'block'
                      }}
                    >
                      {selectedConversation.user.firstName} {selectedConversation.user.lastName}
                    </Text>
                    <Text 
                      style={{ 
                        ...textStyle.caption,
                        color: colors.light.textSecondary
                      }}
                    >
                      {selectedConversation.user.isOnline 
                        ? 'ออนไลน์' 
                        : selectedConversation.user.lastSeen 
                          ? `ออนไลน์ล่าสุด ${dayjs(selectedConversation.user.lastSeen).fromNow()}`
                          : 'ออฟไลน์'
                      }
                    </Text>
                  </div>
                </div>

                <Space>
                  <Button
                    type="text"
                    icon={<PhoneOutlined style={{ color: colors.light.textSecondary, fontSize: '16px' }} />}
                    style={{ 
                      borderRadius: '8px',
                      border: 'none',
                      backgroundColor: 'transparent'
                    }}
                  />
                  <Button
                    type="text"
                    icon={<VideoCameraOutlined style={{ color: colors.light.textSecondary, fontSize: '16px' }} />}
                    style={{ 
                      borderRadius: '8px',
                      border: 'none',
                      backgroundColor: 'transparent'
                    }}
                  />
                  <Button
                    type="text"
                    icon={<MoreOutlined style={{ color: colors.light.textSecondary, fontSize: '16px' }} />}
                    style={{ 
                      borderRadius: '8px',
                      border: 'none',
                      backgroundColor: 'transparent'
                    }}
                  />
                </Space>
              </div>

              <div style={{ 
                flex: 1, 
                padding: '20px',
                overflow: 'auto',
                backgroundColor: colors.light.bgLayout
              }}>
                {messagesLoading ? (
                  <div style={{ textAlign: 'center', margin: '40px 0' }}>
                    <Spin />
                  </div>
                ) : messages.length === 0 ? (
                  <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description="ยังไม่มีข้อความ"
                    style={{ margin: '60px 0' }}
                  />
                ) : (
                  <>
                    {messages.map((message) => (
                      <MessageItem key={message.id} message={message} />
                    ))}
                    <div ref={messagesEndRef} />
                  </>
                )}
              </div>

              <div style={{ 
                padding: '20px',
                borderTop: `1px solid ${colors.light.border}`,
                backgroundColor: colors.light.bgContainer
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Button
                    type="text"
                    icon={<SmileOutlined style={{ color: colors.light.textSecondary, fontSize: '16px' }} />}
                    style={{ 
                      borderRadius: '8px',
                      border: 'none',
                      backgroundColor: 'transparent'
                    }}
                  />
                  
                  <Input.TextArea
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="พิมพ์ข้อความ..."
                    autoSize={{ minRows: 1, maxRows: 4 }}
                    style={{ 
                      flex: 1,
                      borderRadius: '20px',
                      resize: 'none'
                    }}
                  />
                  
                  <Button
                    type="primary"
                    icon={sending ? <LoadingOutlined /> : <SendOutlined />}
                    onClick={sendMessage}
                    disabled={!newMessage.trim() || sending}
                    style={{
                      borderRadius: '50%',
                      width: '44px',
                      height: '44px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: colors.light.primary,
                      borderColor: colors.light.primary
                    }}
                  />
                </div>
              </div>
            </>
          ) : (
            <div style={{ 
              flex: 1, 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center', 
              justifyContent: 'center',
              padding: '40px'
            }}>
              <MessageOutlined style={{ fontSize: '64px', color: colors.light.textTertiary, marginBottom: '16px' }} />
              <Title level={3} style={{ color: colors.light.textSecondary, textAlign: 'center' }}>
                เลือกการสนทนาเพื่อเริ่มแชท
              </Title>
              <Text style={{ color: colors.light.textTertiary, textAlign: 'center' }}>
                เลือกจากรายการการสนทนาทางซ้ายเพื่อดูข้อความและตอบกลับ
              </Text>
            </div>
          )}
        </Card>
      </div>
    </AppLayout>
  );
};

export default ChatPage;
