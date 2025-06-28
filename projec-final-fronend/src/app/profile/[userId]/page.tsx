'use client';

import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Avatar, 
  Typography, 
  Button, 
  Space,
  Row,
  Col,
  Spin,
  message,
  Divider,
  Empty,
  Tag,
  Modal,
  Tooltip
} from 'antd';
import { 
  UserOutlined, 
  MessageOutlined,
  UserAddOutlined,
  UserDeleteOutlined,
  CheckOutlined,
  ClockCircleOutlined,
  HeartOutlined,
  CommentOutlined,
  ShareAltOutlined,
  ArrowLeftOutlined
} from '@ant-design/icons';
import { useRouter, useParams } from 'next/navigation';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../../components/layout/AppLayout';
import { cardStyle, colors, buttonStyle } from '../../../styles/theme';

dayjs.extend(relativeTime);

const { Title, Text, Paragraph } = Typography;

interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  profilePicture?: string;
  bio?: string;
  createdAt: string;
}

interface Post {
  id: string;
  userId: string;
  content: string;
  imageUrl?: string;
  createdAt: string;
  _count: {
    likes: number;
    comments: number;
  };
}

interface FriendshipStatus {
  status: 'none' | 'pending' | 'accepted' | 'blocked';
  isRequestSent: boolean;
  isRequestReceived: boolean;
}

const UserProfilePage: React.FC = () => {
  const router = useRouter();
  const params = useParams();
  const userId = params?.userId as string;
  
  const [user, setUser] = useState<User | null>(null);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [friendshipStatus, setFriendshipStatus] = useState<FriendshipStatus>({
    status: 'none',
    isRequestSent: false,
    isRequestReceived: false
  });

  useEffect(() => {
    if (userId) {
      fetchCurrentUser();
      fetchUser();
      fetchUserPosts();
      fetchFriendshipStatus();
    }
  }, [userId]);

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
        
        // If viewing own profile, redirect to main profile page
        if (userData.id === userId) {
          router.push('/profile');
          return;
        }
      } else {
        router.push('/login');
      }
    } catch (error) {
      console.error('Error fetching current user:', error);
      router.push('/login');
    }
  };

  const fetchUser = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/users/${userId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        message.error('ไม่พบผู้ใช้ที่คุณต้องการดู');
        router.push('/feed');
      }
    } catch (error) {
      console.error('Error fetching user:', error);
      message.error('เกิดข้อผิดพลาดในการโหลดข้อมูลผู้ใช้');
      router.push('/feed');
    }
  };

  const fetchUserPosts = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/posts?userId=${userId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const postsData = await response.json();
        setPosts(postsData);
      }
    } catch (error) {
      console.error('Error fetching user posts:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchFriendshipStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/friends/status/${userId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const statusData = await response.json();
        setFriendshipStatus(statusData);
      }
    } catch (error) {
      console.error('Error fetching friendship status:', error);
    }
  };

  const handleSendFriendRequest = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/friends/requests', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ receiverId: userId }),
      });

      if (response.ok) {
        message.success('ส่งคำขอเป็นเพื่อนแล้ว!');
        fetchFriendshipStatus();
      } else {
        message.error('ไม่สามารถส่งคำขอเป็นเพื่อนได้');
      }
    } catch (error) {
      console.error('Error sending friend request:', error);
      message.error('เกิดข้อผิดพลาดในการส่งคำขอเป็นเพื่อน');
    }
  };

  const handleAcceptFriendRequest = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/friends/requests/accept', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ senderId: userId }),
      });

      if (response.ok) {
        message.success('ยอมรับคำขอเป็นเพื่อนแล้ว!');
        fetchFriendshipStatus();
      } else {
        message.error('ไม่สามารถยอมรับคำขอได้');
      }
    } catch (error) {
      console.error('Error accepting friend request:', error);
      message.error('เกิดข้อผิดพลาดในการยอมรับคำขอ');
    }
  };

  const handleStartChat = () => {
    router.push(`/chat?user=${userId}`);
  };

  const renderActionButton = () => {
    if (friendshipStatus.status === 'accepted') {
      return (
        <Space>
          <Button 
            type="primary" 
            icon={<MessageOutlined />}
            onClick={handleStartChat}
            style={buttonStyle.primary}
          >
            ส่งข้อความ
          </Button>
          <Button 
            icon={<UserDeleteOutlined />}
            onClick={() => message.info('ฟีเจอร์นี้กำลังพัฒนา')}
          >
            ยกเลิกเป็นเพื่อน
          </Button>
        </Space>
      );
    }

    if (friendshipStatus.isRequestReceived) {
      return (
        <Button 
          type="primary" 
          icon={<CheckOutlined />}
          onClick={handleAcceptFriendRequest}
          style={buttonStyle.primary}
        >
          ยอมรับคำขอเป็นเพื่อน
        </Button>
      );
    }

    if (friendshipStatus.isRequestSent) {
      return (
        <Button 
          icon={<ClockCircleOutlined />}
          disabled
        >
          รอการตอบรับ
        </Button>
      );
    }

    return (
      <Button 
        type="primary" 
        icon={<UserAddOutlined />}
        onClick={handleSendFriendRequest}
        style={buttonStyle.primary}
      >
        เพิ่มเป็นเพื่อน
      </Button>
    );
  };

  if (loading || !user) {
    return (
      <AppLayout>
        <div className="flex justify-center items-center min-h-[400px]">
          <Spin size="large" />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Back Button */}
        <Button 
          icon={<ArrowLeftOutlined />}
          onClick={() => router.back()}
          className="mb-4"
        >
          กลับ
        </Button>

        {/* Profile Header */}
        <Card style={cardStyle} className="shadow-sm">
          <Row gutter={[24, 24]} align="middle">
            <Col xs={24} sm={8} className="text-center">
              <Avatar
                size={120}
                src={user.profilePicture}
                icon={<UserOutlined />}
                className="mb-4"
              />
              {renderActionButton()}
            </Col>
            
            <Col xs={24} sm={16}>
              <div className="space-y-4">
                <div>
                  <Title level={2} className="mb-1">
                    {user.firstName} {user.lastName}
                  </Title>
                  <Text className="text-gray-500 text-lg">@{user.username}</Text>
                </div>
                
                {user.bio && (
                  <Paragraph className="text-gray-700 text-base">
                    {user.bio}
                  </Paragraph>
                )}
                
                <div className="flex items-center space-x-1 text-gray-500">
                  <ClockCircleOutlined />
                  <Text>เข้าร่วมเมื่อ {dayjs(user.createdAt).format('MMMM YYYY')}</Text>
                </div>
              </div>
            </Col>
          </Row>
        </Card>

        {/* Posts Section */}
        <Card 
          title={
            <div className="flex items-center space-x-2">
              <Text strong className="text-lg">โพสต์ของ {user.firstName}</Text>
              <Tag color="blue">{posts.length} โพสต์</Tag>
            </div>
          }
          style={cardStyle}
          className="shadow-sm"
        >
          {posts.length > 0 ? (
            <div className="space-y-4">
              {posts.map((post) => (
                <div key={post.id} className="border-b border-gray-100 pb-4 last:border-b-0">
                  <div className="mb-3">
                    <Paragraph className="text-gray-800 mb-2 whitespace-pre-wrap">
                      {post.content}
                    </Paragraph>
                    
                    {post.imageUrl && (
                      <div className="rounded-lg overflow-hidden bg-gray-100 mt-3">
                        <img 
                          src={post.imageUrl} 
                          alt="Post"
                          className="w-full h-auto max-h-64 object-cover"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                          }}
                        />
                      </div>
                    )}
                  </div>
                  
                  <div className="flex items-center justify-between text-gray-500">
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-1">
                        <HeartOutlined />
                        <span>{post._count.likes}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <CommentOutlined />
                        <span>{post._count.comments}</span>
                      </div>
                    </div>
                    <Text className="text-sm">
                      {dayjs(post.createdAt).fromNow()}
                    </Text>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Empty
              description={`${user.firstName} ยังไม่มีโพสต์`}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          )}
        </Card>
      </div>
    </AppLayout>
  );
};

export default UserProfilePage;
