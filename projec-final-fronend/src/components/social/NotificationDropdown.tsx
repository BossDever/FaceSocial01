'use client';

import React, { useState, useEffect } from 'react';
import {
  Dropdown,
  Badge,
  List,
  Avatar,
  Button,
  Typography,
  Space,
  Divider,
  Empty,
  message
} from 'antd';
import {
  BellOutlined,
  UserAddOutlined,
  HeartOutlined,
  MessageOutlined,
  TagOutlined,
  CheckOutlined,
  CloseOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import 'dayjs/locale/th';

dayjs.extend(relativeTime);
dayjs.locale('th');

const { Text } = Typography;

interface Notification {
  id: string;
  type: 'FRIEND_REQUEST' | 'FRIEND_ACCEPTED' | 'POST_LIKE' | 'POST_COMMENT' | 'FACE_TAG' | 'MENTION' | 'SYSTEM';
  title: string;
  message: string;
  data?: any;
  isRead: boolean;
  createdAt: string;
  sender?: {
    id: string;
    username: string;
    firstName: string;
    lastName: string;
    profileImageUrl?: string;
  };
}

const NotificationDropdown: React.FC = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(false);
  const [visible, setVisible] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  const fetchNotifications = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      if (!token) {
        console.log('No token found');
        return;
      }

      const response = await fetch('/api/notifications', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });      if (response.ok) {
        const data = await response.json();
        // API ส่งข้อมูลเป็น array โดยตรง
        setNotifications(Array.isArray(data) ? data : data.notifications || []);
      } else {
        console.error('Failed to fetch notifications:', response.status);
        message.error('ไม่สามารถโหลดการแจ้งเตือนได้');
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
      message.error('เกิดข้อผิดพลาดในการโหลดการแจ้งเตือน');
    } finally {
      setLoading(false);
    }
  };

  const fetchUnreadCount = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch('/api/notifications/unread-count', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setUnreadCount(data.unreadCount || 0);
      }
    } catch (error) {
      console.error('Error fetching unread count:', error);
    }
  };

  const markAsRead = async (notificationIds: string[]) => {
    try {
      const token = localStorage.getItem('token');
      
      await fetch('/api/notifications', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          notificationIds,
          markAsRead: true
        })
      });

      // อัปเดต local state
      setNotifications(prev =>
        prev.map(notif =>
          notificationIds.includes(notif.id)
            ? { ...notif, isRead: true }
            : notif
        )
      );
    } catch (error) {
      console.error('Error marking notifications as read:', error);
    }
  };

  const handleFriendRequest = async (notificationId: string, friendshipId: string, action: 'accept' | 'reject') => {
    try {
      const token = localStorage.getItem('token');
      
      const response = await fetch(`/api/friends/requests/${friendshipId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action })
      });

      if (response.ok) {
        message.success(action === 'accept' ? 'ยอมรับคำขอเป็นเพื่อนแล้ว' : 'ปฏิเสธคำขอเป็นเพื่อนแล้ว');
        markAsRead([notificationId]);
        fetchNotifications(); // รีเฟรชรายการ
      } else {
        message.error('เกิดข้อผิดพลาด');
      }
    } catch (error) {
      console.error('Error handling friend request:', error);
      message.error('เกิดข้อผิดพลาดในการเชื่อมต่อ');
    }
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'FRIEND_REQUEST':
        return <UserAddOutlined className="text-blue-500" />;
      case 'POST_LIKE':
        return <HeartOutlined className="text-red-500" />;
      case 'POST_COMMENT':
        return <MessageOutlined className="text-green-500" />;
      case 'FACE_TAG':
        return <TagOutlined className="text-purple-500" />;
      default:
        return <BellOutlined className="text-gray-500" />;
    }
  };

  const renderNotificationActions = (notification: Notification) => {
    if (notification.type === 'FRIEND_REQUEST' && notification.data?.friendshipId) {
      return (
        <Space>
          <Button
            size="small"
            type="primary"
            icon={<CheckOutlined />}
            onClick={() => handleFriendRequest(notification.id, notification.data.friendshipId, 'accept')}
          >
            ยอมรับ
          </Button>
          <Button
            size="small"
            icon={<CloseOutlined />}
            onClick={() => handleFriendRequest(notification.id, notification.data.friendshipId, 'reject')}
          >
            ปฏิเสธ
          </Button>
        </Space>
      );
    }
    return null;
  };

  useEffect(() => {
    if (visible) {
      fetchNotifications();
      fetchUnreadCount();
    }
  }, [visible]);

  const notificationsList = (
    <div className="w-80 max-h-96 overflow-y-auto">
      <div className="p-3 border-b">
        <div className="flex justify-between items-center">
          <Text strong>การแจ้งเตือน</Text>
          {unreadCount > 0 && (
            <Button
              type="link"
              size="small"
              onClick={() => {
                const unreadIds = notifications.filter(n => !n.isRead).map(n => n.id);
                markAsRead(unreadIds);
              }}
            >
              อ่านทั้งหมด
            </Button>
          )}
        </div>
      </div>

      {notifications.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="ไม่มีการแจ้งเตือน"
          className="py-4"
        />
      ) : (
        <List
          dataSource={notifications}
          renderItem={(notification) => (
            <List.Item
              className={`${!notification.isRead ? 'bg-blue-50' : ''} hover:bg-gray-50 cursor-pointer`}
              onClick={() => {
                if (!notification.isRead) {
                  markAsRead([notification.id]);
                }
              }}
            >
              <List.Item.Meta
                avatar={
                  notification.sender ? (
                    <Avatar src={notification.sender.profileImageUrl} icon={<UserAddOutlined />} />
                  ) : (
                    <Avatar icon={getNotificationIcon(notification.type)} />
                  )
                }
                title={
                  <div className="flex justify-between items-start">
                    <div>
                      <Text strong={!notification.isRead}>
                        {notification.title}
                      </Text>
                      {!notification.isRead && (
                        <Badge status="processing" className="ml-2" />
                      )}
                    </div>                    <Text type="secondary" className="text-xs">
                      {dayjs(notification.createdAt).fromNow()}
                    </Text>
                  </div>
                }
                description={
                  <div className="space-y-2">
                    <div>
                      {notification.sender && (
                        <Text type="secondary">
                          {notification.sender.firstName} {notification.sender.lastName}{' '}
                        </Text>
                      )}
                      <Text type="secondary">{notification.message}</Text>
                    </div>
                    {renderNotificationActions(notification)}
                  </div>
                }
              />
            </List.Item>
          )}
        />
      )}
    </div>
  );

  return (
    <Dropdown
      overlay={notificationsList}
      trigger={['click']}
      placement="bottomRight"
      open={visible}
      onOpenChange={setVisible}
    >
      <Badge count={unreadCount} size="small">
        <Button
          type="text"
          icon={<BellOutlined />}
          className="flex items-center justify-center"
        />
      </Badge>
    </Dropdown>
  );
};

export default NotificationDropdown;
