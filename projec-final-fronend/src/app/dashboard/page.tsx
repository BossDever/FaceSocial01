'use client';

import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Avatar, 
  Typography, 
  Button, 
  Space,
  Statistic,
  Row,
  Col,
  Alert,
  Progress,
  Badge,
  List,
  Divider
} from 'antd';
import { 
  UserOutlined, 
  CameraOutlined,
  SafetyOutlined,
  TeamOutlined,
  HeartOutlined,
  MessageOutlined,
  TrophyOutlined,
  FireOutlined,
  EyeOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import AppLayout from '../../components/layout/AppLayout';
import { cardStyle, colors, buttonStyle, textStyle } from '../../styles/theme';

const { Title, Text, Paragraph } = Typography;

interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  fullName: string;
  isVerified: boolean;
  profilePicture?: string;
}

interface DashboardStats {
  totalPosts: number;
  totalLikes: number;
  totalComments: number;
  totalFriends: number;
  totalMessages: number;
}

interface Activity {
  id: string;
  icon: string;
  text: string;
  time: string;
  isRead: boolean;
}

const DashboardPage: React.FC = () => {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [stats, setStats] = useState<DashboardStats>({
    totalPosts: 0,
    totalLikes: 0,
    totalComments: 0,
    totalFriends: 0,
    totalMessages: 0
  });
  const [recentActivities, setRecentActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchDashboardStats = async (token: string) => {
    try {
      const response = await fetch('/api/dashboard/stats', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setStats(data);
      } else {
        console.error('Failed to fetch dashboard stats');
      }
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    }
  };

  const fetchRecentActivities = async (token: string) => {
    try {
      const response = await fetch('/api/dashboard/activities', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setRecentActivities(data);
      } else {
        console.error('Failed to fetch recent activities');
      }
    } catch (error) {
      console.error('Error fetching recent activities:', error);
    }
  };  useEffect(() => {
    const fetchData = async () => {
      // Check if user is logged in
      const token = localStorage.getItem('token');

      if (!token) {
        router.push('/login');
        return;
      }

      try {
        // Fetch current user data from API
        const userResponse = await fetch('/api/auth/me', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (userResponse.ok) {
          const userData = await userResponse.json();
          setUser(userData);
        } else {
          console.error('Failed to fetch user data');
          router.push('/login');
          return;
        }
        
        // Fetch real stats from API
        await fetchDashboardStats(token);
        
        // Fetch recent activities
        await fetchRecentActivities(token);
      } catch (err) {
        console.error('Error fetching data:', err);
        router.push('/login');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [router]);

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <Text className="text-gray-600">กำลังโหลด...</Text>
          </div>
        </div>
      </AppLayout>
    );
  }

  if (!user) {
    return null;
  }
  const quickActions = [
    {
      icon: <CameraOutlined className="text-2xl text-blue-600" />,
      title: 'โพสต์ใหม่',
      description: 'แชร์ภาพและเรื่องราว',
      action: () => router.push('/feed'),
      color: 'bg-blue-50'
    },
    {
      icon: <MessageOutlined className="text-2xl text-green-600" />,
      title: 'ส่งข้อความ',
      description: 'แชทกับเพื่อนๆ',
      action: () => router.push('/chat'),
      color: 'bg-green-50'
    },
    {
      icon: <UserOutlined className="text-2xl text-purple-600" />,
      title: 'แก้ไขโปรไฟล์',
      description: 'อัปเดตข้อมูลส่วนตัว',
      action: () => router.push('/profile'),
      color: 'bg-purple-50'
    }
  ];

  const getActivityIcon = (iconType: string) => {
    switch (iconType) {
      case 'heart':
        return <HeartOutlined className="text-red-500" />;
      case 'message':
        return <MessageOutlined className="text-blue-500" />;
      case 'team':
      case 'user-add':
        return <TeamOutlined className="text-green-500" />;
      case 'camera':
        return <CameraOutlined className="text-purple-500" />;
      default:
        return <UserOutlined className="text-gray-500" />;
    }
  };

  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Welcome Section */}
        <Card style={cardStyle} className="bg-gradient-to-r from-blue-50 to-purple-50 border-0">
          <Row gutter={24} align="middle">
            <Col xs={24} md={16}>
              <div className="flex items-center space-x-4">                <Avatar 
                  size={80} 
                  src={user.profilePicture} 
                  icon={<UserOutlined />}
                  className="border-4 border-white shadow-lg"
                />
                <div>
                  <Title level={2} className="mb-2 text-gray-800">
                    สวัสดี, {user.firstName} {user.lastName}! 👋
                  </Title>
                  <Text className="text-gray-600 text-lg">
                    ยินดีต้อนรับสู่ FaceSocial - เชื่อมต่อกับเพื่อนๆ และแชร์เรื่องราวของคุณ
                  </Text>
                  <div className="mt-3">
                    <Badge 
                      status={user.isVerified ? "success" : "warning"} 
                      text={user.isVerified ? "ยืนยันตัวตนแล้ว" : "รอการยืนยันตัวตน"} 
                      className="text-sm"
                    />
                  </div>
                </div>
              </div>
            </Col>
            <Col xs={24} md={8} className="text-center">
              <div className="bg-white/50 rounded-lg p-4">
                <SafetyOutlined className="text-4xl text-green-600 mb-2" />
                <Text className="block text-gray-600">
                  ระบบรักษาความปลอดภัย
                </Text>
                <Text strong className="text-green-600">
                  ใช้งานได้ปกติ
                </Text>
              </div>
            </Col>
          </Row>
        </Card>

        {/* Stats Cards */}
        <Row gutter={[16, 16]}>
          <Col xs={12} sm={8} lg={4}>            <Card style={cardStyle} className="text-center hover:shadow-lg transition-shadow">
              <Statistic
                title={<Text className="text-gray-600">โพสต์ทั้งหมด</Text>}
                value={stats.totalPosts}
                prefix={<FireOutlined className="text-orange-500" />}
                valueStyle={{ color: colors.light.primary, fontSize: '24px', fontWeight: 'bold' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} lg={4}>
            <Card style={cardStyle} className="text-center hover:shadow-lg transition-shadow">
              <Statistic
                title={<Text className="text-gray-600">ไลค์ที่ได้รับ</Text>}
                value={stats.totalLikes}
                prefix={<HeartOutlined className="text-red-500" />}
                valueStyle={{ color: colors.light.error, fontSize: '24px', fontWeight: 'bold' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} lg={4}>
            <Card style={cardStyle} className="text-center hover:shadow-lg transition-shadow">
              <Statistic
                title={<Text className="text-gray-600">คอมเมนต์</Text>}
                value={stats.totalComments}
                prefix={<MessageOutlined className="text-blue-500" />}
                valueStyle={{ color: colors.light.primary, fontSize: '24px', fontWeight: 'bold' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} lg={4}>
            <Card style={cardStyle} className="text-center hover:shadow-lg transition-shadow">
              <Statistic
                title={<Text className="text-gray-600">เพื่อน</Text>}
                value={stats.totalFriends}
                prefix={<TeamOutlined className="text-green-500" />}
                valueStyle={{ color: colors.light.success, fontSize: '24px', fontWeight: 'bold' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} lg={4}>
            <Card style={cardStyle} className="text-center hover:shadow-lg transition-shadow">
              <Statistic
                title={<Text className="text-gray-600">ข้อความ</Text>}
                value={stats.totalMessages}
                prefix={<MessageOutlined className="text-purple-500" />}
                valueStyle={{ color: colors.light.info, fontSize: '24px', fontWeight: 'bold' }}
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={[24, 24]}>
          {/* Quick Actions */}
          <Col xs={24} lg={12}>
            <Card 
              title={
                <div className="flex items-center space-x-2">
                  <TrophyOutlined className="text-yellow-500" />
                  <Text strong>การดำเนินการด่วน</Text>
                </div>
              }
              style={cardStyle}
              className="h-full"
            >
              <Space direction="vertical" size={16} className="w-full">
                {quickActions.map((action, index) => (
                  <Card 
                    key={index}
                    size="small" 
                    className={`${action.color} border-0 cursor-pointer hover:shadow-md transition-all duration-200 hover:scale-105`}
                    onClick={action.action}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {action.icon}
                      </div>
                      <div className="flex-1">
                        <Text strong className="block text-gray-800">
                          {action.title}
                        </Text>
                        <Text className="text-gray-600 text-sm">
                          {action.description}
                        </Text>
                      </div>
                    </div>
                  </Card>
                ))}
              </Space>
            </Card>
          </Col>

          {/* Recent Activities */}
          <Col xs={24} lg={12}>
            <Card 
              title={
                <div className="flex items-center space-x-2">
                  <EyeOutlined className="text-blue-500" />
                  <Text strong>กิจกรรมล่าสุด</Text>
                </div>
              }
              style={cardStyle}
              className="h-full"
            >              {recentActivities.length > 0 ? (
                <List
                  dataSource={recentActivities}
                  renderItem={(item) => (
                    <List.Item className="border-0 px-0">
                      <List.Item.Meta
                        avatar={
                          <div className="flex items-center justify-center w-10 h-10 bg-gray-100 rounded-full">
                            {getActivityIcon(item.icon)}
                          </div>
                        }
                        title={<Text className="text-gray-800">{item.text}</Text>}
                        description={<Text className="text-gray-500 text-sm">{item.time}</Text>}
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <div className="text-center py-8">
                  <EyeOutlined className="text-4xl text-gray-300 mb-4" />
                  <Text className="text-gray-500">ยังไม่มีกิจกรรมล่าสุด</Text>
                  <div className="mt-2">
                    <Text className="text-gray-400 text-sm">
                      เริ่มต้นด้วยการโพสต์หรือแชทกับเพื่อนๆ!
                    </Text>
                  </div>
                </div>
              )}
              <Divider className="my-4" />
              <div className="text-center">
                <Button 
                  type="link" 
                  className="text-blue-600 hover:text-blue-700"
                  onClick={() => router.push('/feed')}
                >
                  ดูกิจกรรมทั้งหมด →
                </Button>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Progress Section */}
        <Card 
          title={
            <div className="flex items-center space-x-2">
              <TrophyOutlined className="text-yellow-500" />
              <Text strong>ความก้าวหน้าของคุณ</Text>
            </div>
          }
          style={cardStyle}
        >
          <Row gutter={[24, 24]}>
            <Col xs={24} md={8}>
              <div className="text-center">
                <Text className="block mb-2 text-gray-600">ระดับกิจกรรม</Text>
                <Progress 
                  type="circle" 
                  percent={75} 
                  strokeColor={colors.light.primary}
                  format={(percent) => `${percent}%`}
                />
                <Text className="block mt-2 text-sm text-gray-500">
                  โพสต์อีก 3 ครั้งเพื่อถึงระดับถัดไป
                </Text>
              </div>
            </Col>
            <Col xs={24} md={8}>
              <div className="text-center">
                <Text className="block mb-2 text-gray-600">การมีส่วนร่วม</Text>
                <Progress 
                  type="circle" 
                  percent={60} 
                  strokeColor={colors.light.success}
                  format={(percent) => `${percent}%`}
                />
                <Text className="block mt-2 text-sm text-gray-500">
                  แสดงความคิดเห็นเพิ่มเติม
                </Text>
              </div>
            </Col>
            <Col xs={24} md={8}>
              <div className="text-center">
                <Text className="block mb-2 text-gray-600">ความนิยม</Text>
                <Progress 
                  type="circle" 
                  percent={85} 
                  strokeColor={colors.light.error}
                  format={(percent) => `${percent}%`}
                />
                <Text className="block mt-2 text-sm text-gray-500">
                  โพสต์ของคุณได้รับความนิยมมาก!
                </Text>
              </div>
            </Col>
          </Row>
        </Card>
      </div>
    </AppLayout>
  );
};

export default DashboardPage;
