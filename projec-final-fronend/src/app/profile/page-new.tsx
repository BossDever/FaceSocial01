'use client';

import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Avatar, 
  Typography, 
  Button, 
  Form,
  Input,
  Row,
  Col,
  Space,
  Divider,
  Tabs,
  List,
  Tag,
  Progress,
  Statistic,
  message,
  Modal,
  Upload
} from 'antd';
import { 
  UserOutlined, 
  EditOutlined,
  SaveOutlined,
  CameraOutlined,
  MailOutlined,
  PhoneOutlined,
  CalendarOutlined,
  EnvironmentOutlined,
  LinkOutlined,
  TrophyOutlined,
  HeartOutlined,
  MessageOutlined,
  TeamOutlined,
  PlusOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import type { UploadProps } from 'antd';
import AppLayout from '../../components/layout/AppLayout';
import { cardStyle, colors, buttonStyle } from '../../styles/theme';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  profilePicture?: string;
  bio?: string;
  location?: string;
  website?: string;
  phone?: string;
  dateOfBirth?: string;
  isVerified: boolean;
}

interface UserStats {
  posts: number;
  followers: number;
  following: number;
  likes: number;
}

const ProfilePage: React.FC = () => {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [stats, setStats] = useState<UserStats>({
    posts: 0,
    followers: 0,
    following: 0,
    likes: 0
  });
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchCurrentUser();
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
        setUser(userData);
        
        // Mock stats - in real app, fetch from API
        setStats({
          posts: 12,
          followers: 156,
          following: 89,
          likes: 234
        });

        // Set form initial values
        form.setFieldsValue({
          firstName: userData.firstName,
          lastName: userData.lastName,
          bio: userData.bio || '',
          location: userData.location || '',
          website: userData.website || '',
          phone: userData.phone || ''
        });
      } else {
        router.push('/login');
      }
    } catch (error) {
      console.error('Error fetching user:', error);
      router.push('/login');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (values: any) => {
    try {
      setSaving(true);
      // In real app, update user via API
      console.log('Updating user with:', values);
      
      // Update local state
      if (user) {
        setUser({
          ...user,
          ...values
        });
      }
      
      message.success('บันทึกข้อมูลสำเร็จ!');
      setIsEditing(false);
    } catch (error) {
      console.error('Error saving user:', error);
      message.error('เกิดข้อผิดพลาดในการบันทึกข้อมูล');
    } finally {
      setSaving(false);
    }
  };

  const uploadProps: UploadProps = {
    name: 'avatar',
    showUploadList: false,
    beforeUpload: (file) => {
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        message.error('กรุณาเลือกไฟล์รูปภาพเท่านั้น!');
        return false;
      }
      const isLt2M = file.size / 1024 / 1024 < 2;
      if (!isLt2M) {
        message.error('ขนาดไฟล์ต้องน้อยกว่า 2MB!');
        return false;
      }
      // In real app, upload to server
      message.success('อัปโหลดรูปโปรไฟล์สำเร็จ!');
      return false;
    },
  };

  const recentPosts = [
    {
      id: '1',
      content: 'สวัสดีทุกคน! วันนี้อากาศดีมาก 🌞',
      timestamp: '2 ชั่วโมงที่แล้ว',
      likes: 12,
      comments: 3
    },
    {
      id: '2',
      content: 'เพิ่งเสร็จโปรเจคใหม่ รู้สึกดีมาก! 🎉',
      timestamp: '1 วันที่แล้ว',
      likes: 25,
      comments: 8
    },
    {
      id: '3',
      content: 'ขอบคุณทุกคนที่สนับสนุนนะครับ ❤️',
      timestamp: '3 วันที่แล้ว',
      likes: 45,
      comments: 15
    }
  ];

  const achievements = [
    { title: 'ผู้ใช้งานใหม่', description: 'สมัครสมาชิกสำเร็จ', color: 'blue' },
    { title: 'นักเขียน', description: 'โพสต์ครบ 10 ครั้ง', color: 'green' },
    { title: 'คนนิยม', description: 'ได้รับไลค์มากกว่า 100', color: 'orange' },
    { title: 'เพื่อนดี', description: 'มีเพื่อนมากกว่า 50 คน', color: 'purple' }
  ];

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

  const tabItems = [
    {
      key: 'posts',
      label: (
        <span className="flex items-center space-x-2">
          <MessageOutlined />
          <span>โพสต์</span>
        </span>
      ),
      children: (
        <div className="space-y-4">
          {recentPosts.map((post) => (
            <Card key={post.id} size="small" style={cardStyle} className="hover:shadow-md transition-shadow">
              <div className="space-y-3">
                <Paragraph className="mb-0">{post.content}</Paragraph>
                <div className="flex items-center justify-between text-sm text-gray-500">
                  <span>{post.timestamp}</span>
                  <div className="flex items-center space-x-4">
                    <span className="flex items-center space-x-1">
                      <HeartOutlined className="text-red-500" />
                      <span>{post.likes}</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <MessageOutlined />
                      <span>{post.comments}</span>
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )
    },
    {
      key: 'achievements',
      label: (
        <span className="flex items-center space-x-2">
          <TrophyOutlined />
          <span>ความสำเร็จ</span>
        </span>
      ),
      children: (
        <Row gutter={[16, 16]}>
          {achievements.map((achievement, index) => (
            <Col xs={24} sm={12} lg={6} key={index}>
              <Card size="small" style={cardStyle} className="text-center hover:shadow-md transition-shadow">
                <div className="space-y-2">
                  <TrophyOutlined className="text-2xl text-yellow-500" />
                  <Title level={5} className="mb-1">{achievement.title}</Title>
                  <Text className="text-gray-600 text-sm">{achievement.description}</Text>
                  <Tag color={achievement.color} className="mt-2">ปลดล็อคแล้ว</Tag>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      )
    }
  ];

  return (
    <AppLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Profile Header */}
        <Card style={cardStyle} className="overflow-hidden">
          {/* Cover Photo */}
          <div className="h-48 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 relative">
            <div className="absolute inset-0 bg-black/20"></div>
          </div>

          {/* Profile Info */}
          <div className="px-6 pb-6">
            <div className="flex flex-col md:flex-row md:items-end md:justify-between -mt-16 relative z-10">
              <div className="flex flex-col md:flex-row md:items-end md:space-x-6">
                {/* Avatar */}
                <div className="relative">
                  <Avatar 
                    size={120}
                    src={user.profilePicture}
                    icon={<UserOutlined />}
                    className="border-4 border-white shadow-lg"
                  />
                  <Upload {...uploadProps}>
                    <Button
                      type="primary"
                      shape="circle"
                      icon={<CameraOutlined />}
                      size="small"
                      className="absolute bottom-2 right-2"
                      style={buttonStyle.primary}
                    />
                  </Upload>
                </div>

                {/* Basic Info */}
                <div className="mt-4 md:mt-0 md:pb-4">
                  <div className="flex items-center space-x-3 mb-2">
                    <Title level={2} className="mb-0 text-gray-800">
                      {user.firstName} {user.lastName}
                    </Title>
                    {user.isVerified && (
                      <div className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">
                        ✓ ยืนยันแล้ว
                      </div>
                    )}
                  </div>
                  <Text className="text-gray-600 text-lg">@{user.username}</Text>
                  
                  {/* Contact Info */}
                  <div className="flex flex-wrap items-center gap-4 mt-3 text-gray-600">
                    <div className="flex items-center space-x-1">
                      <MailOutlined />
                      <span>{user.email}</span>
                    </div>
                    {user.location && (
                      <div className="flex items-center space-x-1">
                        <EnvironmentOutlined />
                        <span>{user.location}</span>
                      </div>
                    )}
                    {user.website && (
                      <div className="flex items-center space-x-1">
                        <LinkOutlined />
                        <a href={user.website} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700">
                          {user.website}
                        </a>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="mt-4 md:mt-0 md:pb-4">
                <Space>
                  <Button
                    type={isEditing ? "default" : "primary"}
                    icon={isEditing ? <SaveOutlined /> : <EditOutlined />}
                    onClick={() => {
                      if (isEditing) {
                        form.submit();
                      } else {
                        setIsEditing(true);
                      }
                    }}
                    loading={saving}
                    style={!isEditing ? buttonStyle.primary : buttonStyle.secondary}
                  >
                    {isEditing ? 'บันทึก' : 'แก้ไขโปรไฟล์'}
                  </Button>
                  {isEditing && (
                    <Button
                      onClick={() => {
                        setIsEditing(false);
                        form.resetFields();
                      }}
                    >
                      ยกเลิก
                    </Button>
                  )}
                </Space>
              </div>
            </div>

            {/* Bio */}
            {!isEditing && user.bio && (
              <div className="mt-4">
                <Paragraph className="text-gray-700 text-base">
                  {user.bio}
                </Paragraph>
              </div>
            )}
          </div>
        </Card>

        <Row gutter={[24, 24]}>
          {/* Stats Cards */}
          <Col xs={24} lg={8}>
            <Card 
              title={
                <div className="flex items-center space-x-2">
                  <TeamOutlined className="text-blue-500" />
                  <span>สถิติ</span>
                </div>
              }
              style={cardStyle}
            >
              <Row gutter={[16, 16]}>
                <Col xs={12}>
                  <Statistic
                    title="โพสต์"
                    value={stats.posts}
                    valueStyle={{ color: colors.primary[600], fontSize: '24px', fontWeight: 'bold' }}
                  />
                </Col>
                <Col xs={12}>
                  <Statistic
                    title="ไลค์"
                    value={stats.likes}
                    valueStyle={{ color: colors.danger[500], fontSize: '24px', fontWeight: 'bold' }}
                  />
                </Col>
                <Col xs={12}>
                  <Statistic
                    title="ผู้ติดตาม"
                    value={stats.followers}
                    valueStyle={{ color: colors.success[600], fontSize: '24px', fontWeight: 'bold' }}
                  />
                </Col>
                <Col xs={12}>
                  <Statistic
                    title="กำลังติดตาม"
                    value={stats.following}
                    valueStyle={{ color: colors.secondary[600], fontSize: '24px', fontWeight: 'bold' }}
                  />
                </Col>
              </Row>

              <Divider />

              {/* Progress Indicators */}
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Text className="text-gray-600">ระดับกิจกรรม</Text>
                    <Text className="text-blue-600 font-medium">75%</Text>
                  </div>
                  <Progress percent={75} strokeColor={colors.primary[500]} />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Text className="text-gray-600">ความนิยม</Text>
                    <Text className="text-green-600 font-medium">85%</Text>
                  </div>
                  <Progress percent={85} strokeColor={colors.success[500]} />
                </div>
              </div>
            </Card>
          </Col>

          {/* Edit Form or Content */}
          <Col xs={24} lg={16}>
            {isEditing ? (
              <Card 
                title={
                  <div className="flex items-center space-x-2">
                    <EditOutlined className="text-blue-500" />
                    <span>แก้ไขโปรไฟล์</span>
                  </div>
                }
                style={cardStyle}
              >
                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleSave}
                  className="space-y-4"
                >
                  <Row gutter={16}>
                    <Col xs={24} md={12}>
                      <Form.Item
                        label="ชื่อ"
                        name="firstName"
                        rules={[{ required: true, message: 'กรุณากรอกชื่อ' }]}
                      >
                        <Input size="large" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={12}>
                      <Form.Item
                        label="นามสกุล"
                        name="lastName"
                        rules={[{ required: true, message: 'กรุณากรอกนามสกุล' }]}
                      >
                        <Input size="large" />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Form.Item label="เกี่ยวกับฉัน" name="bio">
                    <TextArea 
                      rows={4} 
                      placeholder="เล่าให้ฟังเกี่ยวกับตัวคุณ..."
                      maxLength={500}
                      showCount
                    />
                  </Form.Item>

                  <Row gutter={16}>
                    <Col xs={24} md={12}>
                      <Form.Item label="ที่อยู่" name="location">
                        <Input size="large" placeholder="เช่น กรุงเทพฯ, ประเทศไทย" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={12}>
                      <Form.Item label="เว็บไซต์" name="website">
                        <Input size="large" placeholder="https://..." />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Form.Item label="เบอร์โทรศัพท์" name="phone">
                    <Input size="large" placeholder="08X-XXX-XXXX" />
                  </Form.Item>
                </Form>
              </Card>
            ) : (
              <Card style={cardStyle}>
                <Tabs 
                  items={tabItems}
                  className="custom-tabs"
                />
              </Card>
            )}
          </Col>
        </Row>
      </div>
    </AppLayout>
  );
};

export default ProfilePage;
