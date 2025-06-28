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

// EditForm component to manage form instance separately
const EditForm: React.FC<{
  user: User;
  onSave: (values: any) => Promise<void>;
  onCancel: () => void;
  saving: boolean;
}> = ({ user, onSave, onCancel, saving }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (user) {
      form.setFieldsValue({
        firstName: user.firstName,
        lastName: user.lastName,
        bio: user.bio || '',
        location: user.location || '',
        website: user.website || '',
        phone: user.phone || ''
      });
    }
  }, [user, form]);

  const handleFinish = async (values: any) => {
    await onSave(values);
  };

  return (
    <Card 
      title={
        <div className="flex items-center space-x-2">
          <EditOutlined className="text-blue-500" />
          <span>แก้ไขโปรไฟล์</span>
        </div>
      }
      style={cardStyle}
      extra={
        <Space>
          <Button onClick={onCancel}>
            ยกเลิก
          </Button>
          <Button 
            type="primary" 
            loading={saving}
            onClick={() => form.submit()}
            style={buttonStyle.primary}
            icon={<SaveOutlined />}
          >
            บันทึก
          </Button>
        </Space>
      }
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
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
  );
};

const ProfilePage: React.FC = () => {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [stats, setStats] = useState<UserStats>({
    posts: 0,
    followers: 0,
    following: 0,
    likes: 0  });  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);

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
      const token = localStorage.getItem('token');
      
      const response = await fetch('/api/profile', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });

      if (response.ok) {
        const result = await response.json();
        message.success('บันทึกข้อมูลสำเร็จ!');
        
        // Update user state
        if (result.user) {
          setUser(result.user);
          
          // Update localStorage as well
          const userData = localStorage.getItem('user');
          if (userData) {
            const parsedUser = JSON.parse(userData);
            const updatedUser = { ...parsedUser, ...result.user };
            localStorage.setItem('user', JSON.stringify(updatedUser));
          }
        }
        
        setIsEditing(false);
      } else {
        const error = await response.json();
        message.error(error.message || 'ไม่สามารถบันทึกข้อมูลได้');
      }
    } catch (error) {
      console.error('Error saving user:', error);
      message.error('เกิดข้อผิดพลาดในการบันทึกข้อมูล');
    } finally {
      setSaving(false);
    }
  };  const handleAvatarUpload = async (file: File) => {
    try {
      console.log('🔄 Starting avatar upload...', file.name, file.size);
      setUploadingAvatar(true);
      const token = localStorage.getItem('token');
      
      const formData = new FormData();
      formData.append('avatar', file);

      console.log('📤 Sending upload request...');
      const response = await fetch('/api/profile/avatar', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      console.log('📥 Upload response status:', response.status);
        if (response.ok) {
        const result = await response.json();
        console.log('✅ Upload successful:', result);
        message.success('อัปโหลดรูปโปรไฟล์สำเร็จ!');
        
        // Update user state with new profile picture
        if (user && result.user) {
          const updatedUser = {
            ...user,
            profilePicture: result.user.profilePicture
          };
          setUser(updatedUser);
          
          // Update localStorage as well
          localStorage.setItem('user', JSON.stringify(updatedUser));
          
          console.log('🔄 User state updated with new profile picture:', result.user.profilePicture);
        }
      } else {
        const error = await response.json();
        console.error('❌ Upload failed:', error);
        message.error(error.message || 'ไม่สามารถอัปโหลดรูปได้');
      }
    } catch (error) {
      console.error('💥 Error uploading avatar:', error);
      message.error('เกิดข้อผิดพลาดในการอัปโหลดรูป');
    } finally {
      setUploadingAvatar(false);
    }
  };
  const uploadProps: UploadProps = {
    name: 'avatar',
    showUploadList: false,
    accept: 'image/*',
    beforeUpload: (file) => {
      console.log('🎯 File selected for upload:', file.name, file.type, file.size);
      
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        console.log('❌ File type validation failed:', file.type);
        message.error('กรุณาเลือกไฟล์รูปภาพเท่านั้น!');
        return false;
      }      const isLt10M = file.size / 1024 / 1024 < 10;
      if (!isLt10M) {
        console.log('❌ File size validation failed:', file.size, 'bytes');
        message.error('ขนาดไฟล์ต้องน้อยกว่า 10MB!');
        return false;
      }
      
      console.log('✅ File validation passed, starting upload...');
      // Call our upload function
      handleAvatarUpload(file);
      return false; // Prevent auto upload
    },
    onChange: (info) => {
      console.log('📤 Upload onChange:', info.file.status);
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
                <div className="relative">                  <Avatar 
                    size={120}
                    src={user.profilePicture ? `${user.profilePicture}?t=${Date.now()}` : undefined}
                    icon={<UserOutlined />}
                    className="border-4 border-white shadow-lg"
                  /><Upload {...uploadProps}>
                    <Button
                      type="primary"
                      shape="circle"
                      icon={uploadingAvatar ? null : <CameraOutlined />}
                      size="small"
                      className="absolute bottom-2 right-2"
                      style={buttonStyle.primary}
                      loading={uploadingAvatar}
                      disabled={uploadingAvatar}
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
              </div>              {/* Action Buttons */}
              <div className="mt-4 md:mt-0 md:pb-4">
                <Space>
                  {!isEditing && (
                    <Button
                      type="primary"
                      icon={<EditOutlined />}
                      onClick={() => setIsEditing(true)}
                      style={buttonStyle.primary}
                    >
                      แก้ไขโปรไฟล์
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
                    valueStyle={{ color: colors.light.primary, fontSize: '24px', fontWeight: 'bold' }}
                  />
                </Col>
                <Col xs={12}>
                  <Statistic
                    title="ไลค์"
                    value={stats.likes}
                    valueStyle={{ color: colors.light.error, fontSize: '24px', fontWeight: 'bold' }}
                  />
                </Col>
                <Col xs={12}>
                  <Statistic
                    title="ผู้ติดตาม"
                    value={stats.followers}
                    valueStyle={{ color: colors.light.success, fontSize: '24px', fontWeight: 'bold' }}
                  />
                </Col>
                <Col xs={12}>
                  <Statistic
                    title="กำลังติดตาม"
                    value={stats.following}
                    valueStyle={{ color: colors.light.info, fontSize: '24px', fontWeight: 'bold' }}
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
                  <Progress percent={75} strokeColor={colors.light.primary} />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Text className="text-gray-600">ความนิยม</Text>
                    <Text className="text-green-600 font-medium">85%</Text>
                  </div>
                  <Progress percent={85} strokeColor={colors.light.success} />
                </div>
              </div>
            </Card>
          </Col>          {/* Edit Form or Content */}
          <Col xs={24} lg={16}>
            {isEditing ? (
              <EditForm
                user={user}
                onSave={handleSave}
                onCancel={() => setIsEditing(false)}
                saving={saving}
              />
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
