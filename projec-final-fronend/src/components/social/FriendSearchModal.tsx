'use client';

import React, { useState, useEffect } from 'react';
import {
  Modal,
  Input,
  List,
  Avatar,
  Button,
  Space,
  Typography,
  Empty,
  message,
  Tag
} from 'antd';
import {
  SearchOutlined,
  UserAddOutlined,
  UserOutlined,
  CheckOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  profileImageUrl?: string;
}

interface FriendSearchModalProps {
  visible: boolean;
  onCancel: () => void;
}

const FriendSearchModal: React.FC<FriendSearchModalProps> = ({
  visible,
  onCancel
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [sentRequests, setSentRequests] = useState<Set<string>>(new Set());
  const searchUsers = async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      if (!token) {
        message.error('กรุณาเข้าสู่ระบบก่อน');
        return;
      }
      
      const response = await fetch(`/api/friends/search-users?q=${encodeURIComponent(query)}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        }
      });

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.users || []);
      } else {
        console.error('Failed to search users:', response.status);
        message.error('ไม่สามารถค้นหาผู้ใช้ได้');
      }
    } catch (error) {
      console.error('Error searching users:', error);
      message.error('เกิดข้อผิดพลาดในการค้นหา');
    } finally {
      setLoading(false);
    }
  };

  const sendFriendRequest = async (friendId: string) => {
    try {
      const token = localStorage.getItem('token');
      
      const response = await fetch('/api/friends/requests', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ friendId })
      });

      if (response.ok) {
        message.success('ส่งคำขอเป็นเพื่อนสำเร็จ!');
        setSentRequests(prev => new Set([...prev, friendId]));
      } else {
        const error = await response.json();
        message.error(error.message || 'เกิดข้อผิดพลาดในการส่งคำขอ');
      }
    } catch (error) {
      console.error('Error sending friend request:', error);
      message.error('เกิดข้อผิดพลาดในการเชื่อมต่อ');
    }
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      searchUsers(searchQuery);
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  const handleClose = () => {
    setSearchQuery('');
    setSearchResults([]);
    setSentRequests(new Set());
    onCancel();
  };

  return (
    <Modal
      title="ค้นหาเพื่อน"
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={600}
    >
      <div className="space-y-4">
        <Input
          prefix={<SearchOutlined />}
          placeholder="ค้นหาด้วยชื่อ, นามสกุล, ชื่อผู้ใช้ หรืออีเมล"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          size="large"
        />

        {searchQuery.length >= 2 && (
          <List
            loading={loading}
            dataSource={searchResults}
            locale={{
              emptyText: (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="ไม่พบผู้ใช้ที่ตรงกับการค้นหา"
                />
              )
            }}
            renderItem={(user) => (
              <List.Item
                actions={[
                  sentRequests.has(user.id) ? (
                    <Tag icon={<ClockCircleOutlined />} color="orange">
                      ส่งคำขอแล้ว
                    </Tag>
                  ) : (
                    <Button
                      type="primary"
                      icon={<UserAddOutlined />}
                      onClick={() => sendFriendRequest(user.id)}
                      size="small"
                    >
                      เพิ่มเพื่อน
                    </Button>
                  )
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <Avatar 
                      src={user.profileImageUrl} 
                      icon={<UserOutlined />}
                      size={48}
                    />
                  }
                  title={
                    <div>
                      <Text strong>{user.firstName} {user.lastName}</Text>
                      <Text type="secondary" className="ml-2">@{user.username}</Text>
                    </div>
                  }
                  description={
                    <Space direction="vertical" size={0}>
                      <Text type="secondary">{user.username}</Text>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        )}

        {searchQuery.length < 2 && (
          <div className="text-center py-8">
            <Text type="secondary">
              กรุณาพิมพ์อย่างน้อย 2 ตัวอักษรเพื่อเริ่มค้นหา
            </Text>
          </div>
        )}
      </div>
    </Modal>
  );
};

export default FriendSearchModal;
