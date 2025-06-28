'use client';

import React, { useState, useEffect } from 'react';
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
  Tabs,
  Modal,
  Tag
} from 'antd';
import { 
  UserOutlined, 
  UserAddOutlined,
  CheckOutlined,
  CloseOutlined,
  SearchOutlined,
  TeamOutlined,
  ClockCircleOutlined,
  HeartOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../components/layout/AppLayout';
import FriendSearchModal from '../../components/social/FriendSearchModal';
import { cardStyle, colors, buttonStyle } from '../../styles/theme';

dayjs.extend(relativeTime);

const { Text, Title } = Typography;
const { Search } = Input;
const { TabPane } = Tabs;

interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  profilePicture?: string;
  mutualFriendsCount?: number;
}

interface FriendRequest {
  id: string;
  senderId: string;
  receiverId: string;
  status: 'pending' | 'accepted' | 'rejected';
  createdAt: string;
  sender: User;
  receiver: User;
}

interface Friend {
  id: string;
  user1Id: string;
  user2Id: string;
  createdAt: string;
  user1: User;
  user2: User;
  friend: User; // The friend user (computed)
}

const FriendsPage: React.FC = () => {
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [friends, setFriends] = useState<Friend[]>([]);
  const [sentRequests, setSentRequests] = useState<FriendRequest[]>([]);
  const [receivedRequests, setReceivedRequests] = useState<FriendRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [processingRequest, setProcessingRequest] = useState<string | null>(null);
  const [searchModalVisible, setSearchModalVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('friends');

  useEffect(() => {
    fetchCurrentUser();
    fetchFriends();
    fetchFriendRequests();
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

  const fetchFriends = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch('/api/friends', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const friendsData = await response.json();
        setFriends(friendsData);
      } else {
        console.error('Failed to fetch friends');
      }
    } catch (error) {
      console.error('Error fetching friends:', error);
    }
  };

  const fetchFriendRequests = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch('/api/friends/requests', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const requestsData = await response.json();
        setSentRequests(requestsData.sent || []);
        setReceivedRequests(requestsData.received || []);
      } else {
        console.error('Failed to fetch friend requests');
      }
    } catch (error) {
      console.error('Error fetching friend requests:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptRequest = async (requestId: string) => {
    try {
      setProcessingRequest(requestId);
      const token = localStorage.getItem('token');
      
      const response = await fetch(`/api/friends/requests/${requestId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: 'accept' }),
      });

      if (response.ok) {
        message.success('ยอมรับคำขอเป็นเพื่อนแล้ว!');
        fetchFriends();
        fetchFriendRequests();
      } else {
        message.error('ไม่สามารถยอมรับคำขอได้');
      }
    } catch (error) {
      console.error('Error accepting friend request:', error);
      message.error('เกิดข้อผิดพลาด');
    } finally {
      setProcessingRequest(null);
    }
  };

  const handleRejectRequest = async (requestId: string) => {
    try {
      setProcessingRequest(requestId);
      const token = localStorage.getItem('token');
      
      const response = await fetch(`/api/friends/requests/${requestId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: 'reject' }),
      });

      if (response.ok) {
        message.success('ปฏิเสธคำขอแล้ว');
        fetchFriendRequests();
      } else {
        message.error('ไม่สามารถปฏิเสธคำขอได้');
      }
    } catch (error) {
      console.error('Error rejecting friend request:', error);
      message.error('เกิดข้อผิดพลาด');
    } finally {
      setProcessingRequest(null);
    }
  };

  const handleCancelRequest = async (requestId: string) => {
    try {
      setProcessingRequest(requestId);
      const token = localStorage.getItem('token');
      
      const response = await fetch(`/api/friends/requests/${requestId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        message.success('ยกเลิกคำขอแล้ว');
        fetchFriendRequests();
      } else {
        message.error('ไม่สามารถยกเลิกคำขอได้');
      }
    } catch (error) {
      console.error('Error cancelling friend request:', error);
      message.error('เกิดข้อผิดพลาด');
    } finally {
      setProcessingRequest(null);
    }
  };

  const refreshData = () => {
    fetchFriends();
    fetchFriendRequests();
  };

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <Card style={cardStyle} className="shadow-sm">
          <div className="flex justify-between items-center">
            <div>              <Title level={2} className="mb-1" style={{ color: '#1890ff' }}>
                <TeamOutlined className="mr-2" />
                เพื่อน
              </Title>
              <Text type="secondary">จัดการเพื่อนและคำขอเป็นเพื่อน</Text>
            </div>
            <Button 
              type="primary" 
              icon={<UserAddOutlined />}              onClick={() => setSearchModalVisible(true)}
              className="bg-blue-500 hover:bg-blue-600"
            >
              ค้นหาเพื่อน
            </Button>
          </div>
        </Card>

        {/* Tabs for different sections */}
        <Card style={cardStyle} className="shadow-sm">
          <Tabs 
            activeKey={activeTab} 
            onChange={setActiveTab}
            size="large"
          >
            <TabPane 
              tab={
                <span>
                  <TeamOutlined />
                  เพื่อน ({friends.length})
                </span>
              } 
              key="friends"
            >
              <Spin spinning={loading}>
                {friends.length === 0 ? (
                  <Empty 
                    description="ยังไม่มีเพื่อน"
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                  />
                ) : (
                  <List
                    grid={{ gutter: 16, xs: 1, sm: 2, md: 2, lg: 3, xl: 3 }}
                    dataSource={friends}
                    renderItem={(friendship) => (
                      <List.Item>
                        <Card 
                          size="small"
                          className="hover:shadow-md transition-shadow"
                        >
                          <div className="text-center">
                            <Avatar 
                              src={friendship.friend.profilePicture} 
                              icon={<UserOutlined />}
                              size={64}
                              className="mb-3"
                            />
                            <div>
                              <Text strong>
                                {friendship.friend.firstName} {friendship.friend.lastName}
                              </Text>
                              <br />
                              <Text type="secondary">@{friendship.friend.username}</Text>
                              <br />
                              <Text type="secondary" className="text-xs">
                                เป็นเพื่อนตั้งแต่ {dayjs(friendship.createdAt).format('DD/MM/YYYY')}
                              </Text>
                            </div>
                            <div className="mt-3 space-x-2">
                              <Button 
                                size="small"
                                onClick={() => router.push(`/chat?user=${friendship.friend.id}`)}
                              >
                                ส่งข้อความ
                              </Button>
                              <Button 
                                size="small"
                                onClick={() => router.push(`/profile/${friendship.friend.username}`)}
                              >
                                ดูโปรไฟล์
                              </Button>
                            </div>
                          </div>
                        </Card>
                      </List.Item>
                    )}
                  />
                )}
              </Spin>
            </TabPane>

            <TabPane 
              tab={
                <span>
                  <ClockCircleOutlined />
                  คำขอที่ได้รับ 
                  {receivedRequests.length > 0 && (
                    <Badge count={receivedRequests.length} className="ml-1" />
                  )}
                </span>
              } 
              key="received"
            >
              <Spin spinning={loading}>
                {receivedRequests.length === 0 ? (
                  <Empty 
                    description="ไม่มีคำขอเป็นเพื่อน"
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                  />
                ) : (
                  <List
                    dataSource={receivedRequests}
                    renderItem={(request) => (
                      <List.Item
                        actions={[
                          <Button
                            key="accept"
                            type="primary"
                            icon={<CheckOutlined />}
                            loading={processingRequest === request.id}                            onClick={() => handleAcceptRequest(request.id)}
                            className="bg-blue-500 hover:bg-blue-600"
                          >
                            ยอมรับ
                          </Button>,
                          <Button
                            key="reject"
                            danger
                            icon={<CloseOutlined />}
                            loading={processingRequest === request.id}
                            onClick={() => handleRejectRequest(request.id)}
                          >
                            ปฏิเสธ
                          </Button>
                        ]}
                      >
                        <List.Item.Meta
                          avatar={
                            <Avatar 
                              src={request.sender.profilePicture} 
                              icon={<UserOutlined />}
                              size={48}
                            />
                          }
                          title={
                            <div>
                              <Text strong>
                                {request.sender.firstName} {request.sender.lastName}
                              </Text>
                              <Text type="secondary" className="ml-2">
                                @{request.sender.username}
                              </Text>
                            </div>
                          }
                          description={
                            <Text type="secondary">
                              ส่งคำขอเป็นเพื่อน {dayjs(request.createdAt).fromNow()}
                            </Text>
                          }
                        />
                      </List.Item>
                    )}
                  />
                )}
              </Spin>
            </TabPane>

            <TabPane 
              tab={
                <span>
                  <HeartOutlined />
                  คำขอที่ส่ง ({sentRequests.length})
                </span>
              } 
              key="sent"
            >
              <Spin spinning={loading}>
                {sentRequests.length === 0 ? (
                  <Empty 
                    description="ไม่มีคำขอที่ส่ง"
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                  />
                ) : (
                  <List
                    dataSource={sentRequests}
                    renderItem={(request) => (
                      <List.Item
                        actions={[
                          <Button
                            key="cancel"
                            danger
                            loading={processingRequest === request.id}
                            onClick={() => handleCancelRequest(request.id)}
                          >
                            ยกเลิกคำขอ
                          </Button>
                        ]}
                      >
                        <List.Item.Meta
                          avatar={
                            <Avatar 
                              src={request.receiver.profilePicture} 
                              icon={<UserOutlined />}
                              size={48}
                            />
                          }
                          title={
                            <div>
                              <Text strong>
                                {request.receiver.firstName} {request.receiver.lastName}
                              </Text>
                              <Text type="secondary" className="ml-2">
                                @{request.receiver.username}
                              </Text>
                              <Tag color="orange" className="ml-2">รออนุมัติ</Tag>
                            </div>
                          }
                          description={
                            <Text type="secondary">
                              ส่งคำขอเมื่อ {dayjs(request.createdAt).fromNow()}
                            </Text>
                          }
                        />
                      </List.Item>
                    )}
                  />
                )}
              </Spin>
            </TabPane>
          </Tabs>
        </Card>

        {/* Friend Search Modal */}        <FriendSearchModal
          visible={searchModalVisible}
          onCancel={() => setSearchModalVisible(false)}
        />
      </div>
    </AppLayout>
  );
};

export default FriendsPage;
