'use client';

import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Avatar, 
  Typography, 
  Button, 
  Space,
  List,
  message,
  Empty,
  Spin,
  Badge,
  Tag,
  Row,
  Col,
  Tooltip
} from 'antd';
import { 
  BellOutlined, 
  UserAddOutlined,
  HeartOutlined,
  MessageOutlined,
  TagOutlined,
  CheckOutlined,
  CloseOutlined,
  UserOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../components/layout/AppLayout';
import { cardStyle, colors } from '../../styles/theme';

dayjs.extend(relativeTime);

const { Text, Title } = Typography;

interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  profilePicture?: string;
}

interface Notification {
  id: string;
  userId: string;
  senderId?: string;
  type: 'FRIEND_REQUEST' | 'FRIEND_ACCEPTED' | 'POST_LIKE' | 'POST_COMMENT' | 'FACE_TAG';
  title: string;
  message: string;
  isRead: boolean;
  createdAt: string;
  data?: any;
  sender?: User;
}

const NotificationsPage: React.FC = () => {
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [processingRequest, setProcessingRequest] = useState<string | null>(null);

  useEffect(() => {
    fetchCurrentUser();
    fetchNotifications();
  }, []);

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

  const fetchNotifications = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch('/api/notifications', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const notificationsData = await response.json();
        setNotifications(notificationsData);
      } else {
        console.error('Failed to fetch notifications');
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    } finally {
      setLoading(false);
    }
  };
  const markAsRead = async (notificationId: string) => {
    try {
      const token = localStorage.getItem('token');
      await fetch(`/api/notifications/${notificationId}/read`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      // Update local state
      setNotifications(prev => 
        prev.map(notif => 
          notif.id === notificationId 
            ? { ...notif, isRead: true }
            : notif
        )
      );
    } catch (error) {
      console.error('Error marking notification as read:', error);
    }
  };

  const markAllAsRead = async () => {
    try {
      const token = localStorage.getItem('token');
      await fetch('/api/notifications/mark-all-read', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      setNotifications(prev => 
        prev.map(notif => ({ ...notif, isRead: true }))
      );
      message.success('อ่านการแจ้งเตือนทั้งหมดแล้ว');
    } catch (error) {
      console.error('Error marking all notifications as read:', error);
      message.error('เกิดข้อผิดพลาด');
    }
  };

  const handleFriendRequest = async (notificationId: string, action: 'accept' | 'reject') => {
    try {
      setProcessingRequest(notificationId);
      const notification = notifications.find(n => n.id === notificationId);
      if (!notification?.data?.friendshipId) return;

      const token = localStorage.getItem('token');
      const response = await fetch(`/api/friends/requests/${notification.data.friendshipId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action }),
      });

      if (response.ok) {
        message.success(action === 'accept' ? 'ยอมรับคำขอเป็นเพื่อนแล้ว!' : 'ปฏิเสธคำขอแล้ว');
        markAsRead(notificationId);
        fetchNotifications(); // Refresh to see updated notifications
      } else {
        message.error('ไม่สามารถดำเนินการได้');
      }
    } catch (error) {
      console.error('Error handling friend request:', error);
      message.error('เกิดข้อผิดพลาด');
    } finally {
      setProcessingRequest(null);
    }
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'FRIEND_REQUEST':
        return <UserAddOutlined style={{ color: '#1890ff' }} />;
      case 'FRIEND_ACCEPTED':
        return <UserAddOutlined style={{ color: '#52c41a' }} />;
      case 'POST_LIKE':
        return <HeartOutlined style={{ color: '#eb2f96' }} />;
      case 'POST_COMMENT':
        return <MessageOutlined style={{ color: '#722ed1' }} />;
      case 'FACE_TAG':
        return <TagOutlined style={{ color: '#fa8c16' }} />;
      default:
        return <BellOutlined />;
    }
  };  const getNotificationActions = (notification: Notification) => {
    // แสดงปุ่มสำหรับ friend request ที่ยังไม่ได้ตอบกลับ
    if (notification.type === 'FRIEND_REQUEST' && notification.data?.friendshipId) {
      return [
        <Button
          key="accept"
          type="primary"
          size="small"
          icon={<CheckOutlined />}
          loading={processingRequest === notification.id}
          onClick={(e) => {
            e.stopPropagation(); // ป้องกันการ trigger onClick ของ List.Item
            handleFriendRequest(notification.id, 'accept');
          }}
          className="bg-green-500 hover:bg-green-600 border-green-500"
        >
          ยอมรับ
        </Button>,
        <Button
          key="reject"
          danger
          size="small"
          icon={<CloseOutlined />}
          loading={processingRequest === notification.id}
          onClick={(e) => {
            e.stopPropagation(); // ป้องกันการ trigger onClick ของ List.Item
            handleFriendRequest(notification.id, 'reject');
          }}
        >
          ปฏิเสธ
        </Button>
      ];
    }
    return [];
  };

  const unreadCount = notifications.filter(n => !n.isRead).length;

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <Card style={cardStyle} className="shadow-sm">
          <div className="flex justify-between items-center">
            <div>
              <Title level={2} className="mb-1" style={{ color: '#1890ff' }}>
                <BellOutlined className="mr-2" />
                การแจ้งเตือน
                {unreadCount > 0 && (
                  <Badge count={unreadCount} className="ml-2" />
                )}
              </Title>
              <Text type="secondary">ติดตามกิจกรรมและการแจ้งเตือนต่างๆ</Text>
            </div>
            {unreadCount > 0 && (
              <Button 
                type="primary" 
                onClick={markAllAsRead}
                className="bg-blue-500 hover:bg-blue-600"
              >
                อ่านทั้งหมด
              </Button>
            )}
          </div>
        </Card>

        {/* Notifications List */}
        <Card style={cardStyle} className="shadow-sm">
          <Spin spinning={loading}>
            {notifications.length === 0 ? (
              <Empty 
                description="ไม่มีการแจ้งเตือน"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            ) : (
              <List
                dataSource={notifications}
                renderItem={(notification) => (
                  <List.Item                    className={`${!notification.isRead ? 'bg-blue-50' : ''} hover:bg-gray-50 transition-colors rounded-lg p-4 mb-2`}
                    actions={getNotificationActions(notification)}
                    onClick={() => {
                      if (!notification.isRead) {
                        markAsRead(notification.id);
                      }
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    <List.Item.Meta
                      avatar={
                        <div className="relative">
                          <Avatar 
                            src={notification.sender?.profilePicture} 
                            icon={<UserOutlined />}
                            size={48}
                          />
                          <div className="absolute -bottom-1 -right-1 bg-white rounded-full p-1">
                            {getNotificationIcon(notification.type)}
                          </div>
                        </div>
                      }
                      title={
                        <div className="flex items-center justify-between">                          <div>
                            <Text strong={!notification.isRead}>
                              {notification.title}
                            </Text>
                            {!notification.isRead && (
                              <Badge dot className="ml-2" />
                            )}
                          </div>
                          <Text type="secondary" className="text-sm">
                            {dayjs(notification.createdAt).fromNow()}
                          </Text>
                        </div>
                      }
                      description={
                        <div>
                          <Text type="secondary">
                            {notification.sender && (
                              <span className="font-medium text-blue-600">
                                {notification.sender.firstName} {notification.sender.lastName}
                              </span>
                            )}{' '}
                            {notification.message}
                          </Text>                          <div className="mt-1">
                            <Tag color={notification.isRead ? 'default' : 'blue'}>
                              {notification.type === 'FRIEND_REQUEST' && 'คำขอเป็นเพื่อน'}
                              {notification.type === 'FRIEND_ACCEPTED' && 'ยอมรับเป็นเพื่อน'}
                              {notification.type === 'POST_LIKE' && 'ไลค์โพสต์'}
                              {notification.type === 'POST_COMMENT' && 'แสดงความเห็น'}
                              {notification.type === 'FACE_TAG' && 'แท็กในรูปภาพ'}
                            </Tag>
                          </div>
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            )}
          </Spin>
        </Card>
      </div>
    </AppLayout>
  );
};

export default NotificationsPage;
