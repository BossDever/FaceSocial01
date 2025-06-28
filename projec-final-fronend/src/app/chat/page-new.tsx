'use client';

import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  Avatar, 
  Typography, 
  Button, 
  Space,
  Input,
  List,
  message,
  Empty,
  Spin,
  Row,
  Col,
  Badge,
  Divider,
  Tooltip
} from 'antd';
import { 
  UserOutlined, 
  SendOutlined,
  SearchOutlined,
  TeamOutlined,
  ClockCircleOutlined,
  MessageOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../components/layout/AppLayout';
import { cardStyle, colors, buttonStyle } from '../../styles/theme';

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
  senderId: string;
  receiverId: string;
  content: string;
  createdAt: string;
  read: boolean;
  sender: User;
  receiver: User;
}

interface Conversation {
  id: string;
  user: User;
  lastMessage?: {
    content: string;
    createdAt: string;
  };
  unreadCount: number;
}

const ChatPage: React.FC = () => {
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchCurrentUser();
    fetchConversations();
  }, []);

  useEffect(() => {
    if (selectedConversation) {
      fetchMessages(selectedConversation.user.id);
    }
  }, [selectedConversation]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchCurrentUser = async () => {
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
      } else {
        router.push('/login');
      }
    } catch (error) {
      console.error('Error fetching user:', error);
      router.push('/login');
    }
  };

  const fetchConversations = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch('/api/chat/conversations', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const conversationsData = await response.json();
        setConversations(conversationsData);
      } else {
        console.error('Failed to fetch conversations');
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchMessages = async (userId: string) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch(`/api/chat/messages/${userId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const messagesData = await response.json();
        setMessages(messagesData);
      } else {
        console.error('Failed to fetch messages');
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !selectedConversation || sending) return;

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
          content: newMessage.trim(),
        }),
      });

      if (response.ok) {
        setNewMessage('');
        fetchMessages(selectedConversation.user.id);
        fetchConversations(); // Update conversation list
      } else {
        message.error('ไม่สามารถส่งข้อความได้');
      }
    } catch (error) {
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

  // Mock sample conversations if no conversations available
  const sampleConversations: Conversation[] = [
    {
      id: 'sample-1',
      user: {
        id: 'user-1',
        username: 'alice_wonder',
        firstName: 'Alice',
        lastName: 'Wonder',
        profilePicture: '/images/avatars/alice.jpg',
        isOnline: true
      },
      lastMessage: {
        content: 'สวัสดี! ยินดีที่ได้รู้จัก 😊',
        createdAt: new Date(Date.now() - 30 * 60 * 1000).toISOString()
      },
      unreadCount: 2
    },
    {
      id: 'sample-2',
      user: {
        id: 'user-2',
        username: 'bob_builder',
        firstName: 'Bob',
        lastName: 'Builder',
        profilePicture: '/images/avatars/bob.jpg',
        isOnline: false
      },
      lastMessage: {
        content: 'มาแชทกันเร็วๆ นี้นะ!',
        createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
      },
      unreadCount: 0
    },
    {
      id: 'sample-3',
      user: {
        id: 'user-3',
        username: 'sarah_dev',
        firstName: 'Sarah',
        lastName: 'Developer',
        profilePicture: '/images/avatars/sarah.jpg',
        isOnline: true
      },
      lastMessage: {
        content: 'ระบบใหม่ของเราเจ๋งมาก!',
        createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
      },
      unreadCount: 1
    }
  ];

  const displayConversations = conversations.length > 0 ? conversations : sampleConversations;
  const filteredConversations = displayConversations.filter(conv =>
    conv.user.firstName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    conv.user.lastName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    conv.user.username.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <AppLayout>
      <div className="h-[calc(100vh-180px)]">
        <Row gutter={0} className="h-full">
          {/* Conversations Sidebar */}
          <Col xs={24} md={8} lg={6} className="h-full">
            <Card 
              style={{ ...cardStyle, height: '100%' }} 
              className="shadow-sm border-r border-gray-200"
              bodyStyle={{ padding: 0, height: '100%' }}
            >
              {/* Search Header */}
              <div className="p-4 border-b border-gray-100">
                <div className="flex items-center justify-between mb-3">
                  <Title level={5} className="m-0 flex items-center space-x-2">
                    <MessageOutlined className="text-blue-600" />
                    <span>แชท</span>
                  </Title>
                  <Badge count={filteredConversations.reduce((sum, conv) => sum + conv.unreadCount, 0)}>
                    <TeamOutlined className="text-gray-600 text-lg" />
                  </Badge>
                </div>
                <Search
                  placeholder="ค้นหาการสนทนา..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="rounded-lg"
                />
              </div>

              {/* Conversations List */}
              <div className="flex-1 overflow-y-auto" style={{ height: 'calc(100% - 120px)' }}>
                {loading ? (
                  <div className="flex items-center justify-center h-32">
                    <Spin />
                  </div>
                ) : filteredConversations.length > 0 ? (
                  <List
                    dataSource={filteredConversations}
                    renderItem={(conversation) => (
                      <List.Item
                        className={`cursor-pointer hover:bg-gray-50 transition-colors px-4 py-3 border-0 ${
                          selectedConversation?.id === conversation.id ? 'bg-blue-50 border-r-2 border-blue-500' : ''
                        }`}
                        onClick={() => setSelectedConversation(conversation)}
                      >
                        <List.Item.Meta
                          avatar={
                            <Badge dot={conversation.user.isOnline} color="green">
                              <Avatar 
                                src={conversation.user.profilePicture} 
                                icon={<UserOutlined />}
                                size={48}
                              />
                            </Badge>
                          }
                          title={
                            <div className="flex items-center justify-between">
                              <Text strong className="text-gray-800">
                                {conversation.user.firstName} {conversation.user.lastName}
                              </Text>
                              {conversation.unreadCount > 0 && (
                                <Badge count={conversation.unreadCount} size="small" />
                              )}
                            </div>
                          }
                          description={
                            <div className="space-y-1">
                              <Text className="text-gray-500 text-sm">
                                @{conversation.user.username}
                              </Text>
                              {conversation.lastMessage && (
                                <div className="flex items-center justify-between">
                                  <Text 
                                    className={`text-sm truncate max-w-32 ${
                                      conversation.unreadCount > 0 ? 'font-medium text-gray-800' : 'text-gray-500'
                                    }`}
                                  >
                                    {conversation.lastMessage.content}
                                  </Text>
                                  <Text className="text-xs text-gray-400">
                                    {dayjs(conversation.lastMessage.createdAt).fromNow()}
                                  </Text>
                                </div>
                              )}
                              <div className="flex items-center space-x-1 text-xs text-gray-400">
                                <span className={`w-2 h-2 rounded-full ${conversation.user.isOnline ? 'bg-green-500' : 'bg-gray-300'}`}></span>
                                <span>{conversation.user.isOnline ? 'ออนไลน์' : 'ออฟไลน์'}</span>
                              </div>
                            </div>
                          }
                        />
                      </List.Item>
                    )}
                  />
                ) : (
                  <div className="text-center py-8">
                    <Empty
                      description="ไม่พบการสนทนา"
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                    />
                  </div>
                )}
              </div>
            </Card>
          </Col>

          {/* Chat Area */}
          <Col xs={24} md={16} lg={18} className="h-full">
            {selectedConversation ? (
              <Card 
                style={{ ...cardStyle, height: '100%' }} 
                className="shadow-sm"
                bodyStyle={{ padding: 0, height: '100%', display: 'flex', flexDirection: 'column' }}
              >
                {/* Chat Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-100">
                  <div className="flex items-center space-x-3">
                    <Badge dot={selectedConversation.user.isOnline} color="green">
                      <Avatar 
                        src={selectedConversation.user.profilePicture} 
                        icon={<UserOutlined />}
                        size={48}
                      />
                    </Badge>
                    <div>
                      <Text strong className="text-gray-800 text-lg">
                        {selectedConversation.user.firstName} {selectedConversation.user.lastName}
                      </Text>
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <span>@{selectedConversation.user.username}</span>
                        <span>•</span>
                        <span className={selectedConversation.user.isOnline ? 'text-green-600' : 'text-gray-400'}>
                          {selectedConversation.user.isOnline ? 'ออนไลน์' : 'ออฟไลน์'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
                  {messages.length > 0 ? (
                    messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.senderId === currentUser?.id ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`flex items-end space-x-2 max-w-xs lg:max-w-md ${
                          message.senderId === currentUser?.id ? 'flex-row-reverse space-x-reverse' : 'flex-row'
                        }`}>
                          <Avatar 
                            src={message.sender.profilePicture} 
                            icon={<UserOutlined />}
                            size={32}
                          />
                          <div className={`px-4 py-2 rounded-2xl ${
                            message.senderId === currentUser?.id
                              ? 'bg-blue-600 text-white rounded-br-sm'
                              : 'bg-white text-gray-800 rounded-bl-sm border border-gray-200'
                          }`}>
                            <Text className={message.senderId === currentUser?.id ? 'text-white' : 'text-gray-800'}>
                              {message.content}
                            </Text>
                            <div className={`text-xs mt-1 ${
                              message.senderId === currentUser?.id ? 'text-blue-100' : 'text-gray-500'
                            }`}>
                              {dayjs(message.createdAt).format('HH:mm')}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <MessageOutlined className="text-4xl text-gray-300 mb-2" />
                      <Text className="text-gray-500">เริ่มการสนทนาใหม่</Text>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Message Input */}
                <div className="p-4 border-t border-gray-100 bg-white">
                  <div className="flex items-end space-x-3">
                    <div className="flex-1">
                      <Input.TextArea
                        placeholder="พิมพ์ข้อความ..."
                        value={newMessage}
                        onChange={(e) => setNewMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        rows={1}
                        autoSize={{ minRows: 1, maxRows: 4 }}
                        className="border-gray-200 rounded-lg resize-none"
                      />
                    </div>
                    <Button
                      type="primary"
                      icon={<SendOutlined />}
                      onClick={sendMessage}
                      loading={sending}
                      disabled={!newMessage.trim()}
                      style={{ ...buttonStyle.primary, minWidth: '48px' }}
                      className="h-10"
                    />
                  </div>
                </div>
              </Card>
            ) : (
              <Card 
                style={{ ...cardStyle, height: '100%' }} 
                className="shadow-sm flex items-center justify-center"
              >
                <div className="text-center">
                  <MessageOutlined className="text-6xl text-gray-300 mb-4" />
                  <Title level={4} className="text-gray-500 mb-2">
                    เลือกการสนทนา
                  </Title>
                  <Text className="text-gray-400">
                    เลือกการสนทนาจากรายการเพื่อเริ่มแชท
                  </Text>
                </div>
              </Card>
            )}
          </Col>
        </Row>
      </div>
    </AppLayout>
  );
};

export default ChatPage;
