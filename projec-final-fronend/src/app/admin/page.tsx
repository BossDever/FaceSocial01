'use client';

import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col,
  Typography,
  Statistic,
  Table,
  Button,
  Input,
  Modal,
  Form,
  Select,
  message,
  Tabs,
  Avatar,
  Badge,
  Space,
  Popconfirm,
  Tag,
  Progress,
  Tooltip,
  Alert,
  Divider,
  List,
  Upload,
  DatePicker
} from 'antd';
import { 
  UserOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  SettingOutlined,
  DashboardOutlined,
  TeamOutlined,
  MessageOutlined,
  SafetyOutlined,
  DatabaseOutlined,
  CloudUploadOutlined,
  DownloadOutlined,
  ClearOutlined,
  ToolOutlined,
  FileImageOutlined,
  ReloadOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ScanOutlined,
  FileOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import AppLayout from '../../components/layout/AppLayout';
import { colors, cardStyle } from '../../styles/theme';

const { Title, Text } = Typography;
const { Search } = Input;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  phone?: string;
  isActive: boolean;
  isVerified: boolean;
  profileImageUrl?: string;
  createdAt: string;
  lastLoginAt?: string;
}

interface AdminStats {
  totalUsers: number;
  activeUsers: number;
  totalPosts: number;
  totalMessages: number;
  onlineUsers: number;
}

interface DatabaseStats {
  users: {
    total: number;
    active: number;
    inactive: number;
    verified: number;
    unverified: number;
  };
  chats: {
    total: number;
    today: number;
    thisWeek: number;
    thisMonth: number;
  };
  embeddings: {
    total: number;
    totalSize: number;
    averageSize: number;
    orphaned: number;
  };
  system: {
    dbSize: string;
    lastBackup: string;
    uptime: string;
    performance: {
      avgResponseTime: number;
      totalQueries: number;
      slowQueries: number;
    };
  };
}

interface EmbeddingFile {
  id: string;
  userId: string;
  fileName: string;
  filePath: string;
  size: number;
  format: string;
  quality: string;
  faceCount: number;
  createdAt: string;
  lastUsed: string;
  accuracy: number;
  isActive: boolean;
}

const AdminDashboard: React.FC = () => {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [stats, setStats] = useState<AdminStats>({
    totalUsers: 0,
    activeUsers: 0,
    totalPosts: 0,
    totalMessages: 0,
    onlineUsers: 0
  });
    // User Management State
  const [users, setUsers] = useState<User[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [searchText, setSearchText] = useState('');
  const [form] = Form.useForm();

  // Database Management State
  const [databaseStats, setDatabaseStats] = useState<DatabaseStats | null>(null);
  const [databaseLoading, setDatabaseLoading] = useState(false);
  const [operationLoading, setOperationLoading] = useState(false);
  // Embeddings Management State
  const [embeddings, setEmbeddings] = useState<EmbeddingFile[]>([]);
  const [embeddingsLoading, setEmbeddingsLoading] = useState(false);
  const [embeddingStats, setEmbeddingStats] = useState<any>(null);
  const [selectedEmbeddings, setSelectedEmbeddings] = useState<string[]>([]);
  const [embeddingFilter, setEmbeddingFilter] = useState('all');

  // System Logs State
  const [logs, setLogs] = useState<any[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [logStats, setLogStats] = useState<any>(null);
  const [logFilters, setLogFilters] = useState({
    level: 'all',
    source: 'all',
    search: '',
    startDate: null,
    endDate: null
  });

  useEffect(() => {
    checkAdminAccess();
  }, []);

  const checkAdminAccess = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      // Get current user info
      const userResponse = await fetch('/api/auth/me', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (userResponse.ok) {
        const userData = await userResponse.json();
        setCurrentUser(userData);
        
        // Check if user is admin (you can modify this logic based on your needs)
        if (userData.username !== 'admin01') {
          message.error('ไม่มีสิทธิ์เข้าใช้หน้านี้');
          router.push('/dashboard');
          return;
        }
          // Load admin data
        await loadAdminStats(token);
        await loadUsers(token);
        await loadDatabaseStats();
        await loadEmbeddings();
        await loadLogs();
        await loadLogs();
      } else {
        router.push('/login');
      }
    } catch (error) {
      console.error('Error checking admin access:', error);
      router.push('/login');
    } finally {
      setLoading(false);
    }
  };

  const loadAdminStats = async (token: string) => {
    try {
      const response = await fetch('/api/admin/stats', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error loading admin stats:', error);
    }
  };

  const loadUsers = async (token: string) => {
    setUsersLoading(true);
    try {
      const response = await fetch('/api/admin/users', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setUsers(data);
      } else {
        message.error('ไม่สามารถโหลดข้อมูลผู้ใช้ได้');
      }
    } catch (error) {
      console.error('Error loading users:', error);
      message.error('เกิดข้อผิดพลาดในการโหลดข้อมูล');
    } finally {
      setUsersLoading(false);
    }
  };

  const handleEditUser = (user: User) => {
    setSelectedUser(user);
    form.setFieldsValue(user);
    setEditModalVisible(true);
  };

  const handleSaveUser = async () => {
    try {
      const values = await form.validateFields();
      const token = localStorage.getItem('token');
      
      const response = await fetch(`/api/admin/users/${selectedUser?.id}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(values)
      });

      if (response.ok) {
        message.success('อัพเดทข้อมูลผู้ใช้สำเร็จ');
        setEditModalVisible(false);
        loadUsers(token!);
      } else {
        message.error('ไม่สามารถอัพเดทข้อมูลได้');
      }
    } catch (error) {
      console.error('Error updating user:', error);
      message.error('เกิดข้อผิดพลาดในการอัพเดท');
    }
  };

  const handleDeleteUser = async (userId: string) => {
    try {
      const token = localStorage.getItem('token');
      
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        message.success('ลบผู้ใช้สำเร็จ');
        loadUsers(token!);
      } else {
        message.error('ไม่สามารถลบผู้ใช้ได้');
      }
    } catch (error) {
      console.error('Error deleting user:', error);
      message.error('เกิดข้อผิดพลาดในการลบ');
    }
  };

  const loadDatabaseStats = async () => {
    setDatabaseLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/admin/database', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setDatabaseStats(data);
      } else {
        message.error('ไม่สามารถโหลดสถิติฐานข้อมูลได้');
      }
    } catch (error) {
      console.error('Error loading database stats:', error);
      message.error('เกิดข้อผิดพลาดในการโหลดข้อมูล');
    } finally {
      setDatabaseLoading(false);
    }
  };

  const performDatabaseOperation = async (action: string, table?: string, options?: any) => {
    setOperationLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/admin/database', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action, table, options })
      });

      if (response.ok) {
        const result = await response.json();
        message.success(result.message);
        loadDatabaseStats(); // Refresh stats
      } else {
        const error = await response.json();
        message.error(error.message || 'ไม่สามารถดำเนินการได้');
      }
    } catch (error) {
      console.error('Database operation error:', error);
      message.error('เกิดข้อผิดพลาดในการดำเนินการ');
    } finally {
      setOperationLoading(false);
    }
  };

  const loadEmbeddings = async () => {
    setEmbeddingsLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/admin/embeddings?filter=${embeddingFilter}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setEmbeddings(data.embeddings);
        setEmbeddingStats(data.stats);
      } else {
        message.error('ไม่สามารถโหลดข้อมูล embeddings ได้');
      }
    } catch (error) {
      console.error('Error loading embeddings:', error);
      message.error('เกิดข้อผิดพลาดในการโหลดข้อมูล');
    } finally {
      setEmbeddingsLoading(false);
    }
  };

  const performEmbeddingOperation = async (action: string, embeddingIds?: string[], userId?: string) => {
    setOperationLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/admin/embeddings', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action, embeddingIds, userId })
      });

      if (response.ok) {
        const result = await response.json();
        message.success(result.message);
        loadEmbeddings(); // Refresh data
        setSelectedEmbeddings([]); // Clear selection
      } else {
        const error = await response.json();
        message.error(error.message || 'ไม่สามารถดำเนินการได้');
      }
    } catch (error) {
      console.error('Embedding operation error:', error);
      message.error('เกิดข้อผิดพลาดในการดำเนินการ');
    } finally {
      setOperationLoading(false);
    }
  };
  // System Logs Functions
  const loadLogs = async () => {
    setLogsLoading(true);
    try {
      const token = localStorage.getItem('token');
      const params: any = {
        page: '1',
        pageSize: '50',
        level: logFilters.level,
        source: logFilters.source
      };

      if (logFilters.search) params.search = logFilters.search;
      if (logFilters.startDate) params.startDate = logFilters.startDate;
      if (logFilters.endDate) params.endDate = logFilters.endDate;

      const queryParams = new URLSearchParams(params);

      const response = await fetch(`/api/admin/logs?${queryParams}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs);
        setLogStats(data.stats);
      } else {
        message.error('ไม่สามารถโหลดข้อมูล logs ได้');
      }
    } catch (error) {
      console.error('Error loading logs:', error);
      message.error('เกิดข้อผิดพลาดในการโหลดข้อมูล');
    } finally {
      setLogsLoading(false);
    }
  };

  const performLogOperation = async (action: string, options?: any) => {
    setOperationLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/admin/logs', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action, options })
      });

      if (response.ok) {
        const result = await response.json();
        message.success(result.message);
        if (action === 'clear') {
          loadLogs(); // Refresh data after clearing
        }
      } else {
        const error = await response.json();
        message.error(error.message || 'ไม่สามารถดำเนินการได้');
      }
    } catch (error) {
      console.error('Log operation error:', error);
      message.error('เกิดข้อผิดพลาดในการดำเนินการ');
    } finally {
      setOperationLoading(false);
    }
  };

  const userColumns = [
    {
      title: 'โปรไฟล์',
      key: 'profile',
      render: (record: User) => (
        <Space>
          <Avatar 
            src={record.profileImageUrl}
            icon={<UserOutlined />}
            size={40}
          />
          <div>
            <Text strong>{record.firstName} {record.lastName}</Text>
            <br />
            <Text type="secondary">@{record.username}</Text>
          </div>
        </Space>
      )
    },
    {
      title: 'อีเมล',
      dataIndex: 'email',
      key: 'email'
    },
    {
      title: 'เบอร์โทร',
      dataIndex: 'phone',
      key: 'phone',
      render: (phone: string) => phone || '-'
    },
    {
      title: 'สถานะ',
      key: 'status',
      render: (record: User) => (
        <Space direction="vertical" size="small">
          <Badge 
            status={record.isActive ? 'success' : 'error'} 
            text={record.isActive ? 'ใช้งาน' : 'ระงับ'} 
          />
          <Tag color={record.isVerified ? 'green' : 'orange'}>
            {record.isVerified ? 'ยืนยันแล้ว' : 'รอยืนยัน'}
          </Tag>
        </Space>
      )
    },
    {
      title: 'วันที่สมัคร',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => new Date(date).toLocaleDateString('th-TH')
    },
    {
      title: 'เข้าสู่ระบบล่าสุด',
      dataIndex: 'lastLoginAt',
      key: 'lastLoginAt',
      render: (date: string) => date ? new Date(date).toLocaleDateString('th-TH') : '-'
    },
    {
      title: 'จัดการ',
      key: 'actions',
      render: (record: User) => (
        <Space>
          <Button 
            type="primary" 
            icon={<EditOutlined />}
            size="small"
            onClick={() => handleEditUser(record)}
          >
            แก้ไข
          </Button>
          <Popconfirm
            title="คุณแน่ใจหรือไม่ที่จะลบผู้ใช้นี้?"
            onConfirm={() => handleDeleteUser(record.id)}
            okText="ลบ"
            cancelText="ยกเลิก"
          >
            <Button 
              danger 
              icon={<DeleteOutlined />}
              size="small"
            >
              ลบ
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ];

  const filteredUsers = users.filter(user => 
    user.firstName.toLowerCase().includes(searchText.toLowerCase()) ||
    user.lastName.toLowerCase().includes(searchText.toLowerCase()) ||
    user.username.toLowerCase().includes(searchText.toLowerCase()) ||
    user.email.toLowerCase().includes(searchText.toLowerCase())
  );

  if (loading) {
    return (
      <AppLayout>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '400px' 
        }}>
          <Text>กำลังโหลด...</Text>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div style={{ padding: '24px' }}>
        <Title level={2}>
          <DashboardOutlined /> Admin Dashboard
        </Title>
        
        <Tabs defaultActiveKey="stats">
          {/* Statistics Tab */}
          <TabPane tab={<span><DashboardOutlined />สถิติระบบ</span>} key="stats">
            <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
              <Col xs={24} sm={12} lg={6}>
                <Card style={cardStyle}>
                  <Statistic
                    title="ผู้ใช้ทั้งหมด"
                    value={stats.totalUsers}
                    prefix={<TeamOutlined />}
                    valueStyle={{ color: colors.light.primary }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={12} lg={6}>
                <Card style={cardStyle}>
                  <Statistic
                    title="ผู้ใช้ใช้งาน"
                    value={stats.activeUsers}
                    prefix={<UserOutlined />}
                    valueStyle={{ color: colors.light.success }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={12} lg={6}>
                <Card style={cardStyle}>
                  <Statistic
                    title="โพสต์ทั้งหมด"
                    value={stats.totalPosts}
                    prefix={<MessageOutlined />}
                    valueStyle={{ color: colors.light.warning }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={12} lg={6}>
                <Card style={cardStyle}>
                  <Statistic
                    title="ผู้ใช้ออนไลน์"
                    value={stats.onlineUsers}
                    prefix={<SafetyOutlined />}
                    valueStyle={{ color: colors.light.error }}
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* User Management Tab */}
          <TabPane tab={<span><TeamOutlined />จัดการผู้ใช้</span>} key="users">
            <Card style={cardStyle}>
              <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between' }}>
                <Search
                  placeholder="ค้นหาผู้ใช้..."
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  style={{ width: 300 }}
                />
                <Button 
                  type="primary" 
                  onClick={() => loadUsers(localStorage.getItem('token')!)}
                  loading={usersLoading}
                >
                  รีเฟรช
                </Button>
              </div>
              
              <Table
                columns={userColumns}
                dataSource={filteredUsers}
                rowKey="id"
                loading={usersLoading}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `ทั้งหมด ${total} คน`
                }}
              />
            </Card>
          </TabPane>          {/* Database Management Tab */}
          <TabPane tab={<span><DatabaseOutlined />จัดการฐานข้อมูล</span>} key="database">
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              {/* Database Statistics */}
              <Card style={cardStyle}>
                <Title level={4}>
                  <DatabaseOutlined /> สถิติฐานข้อมูล
                  <Button 
                    style={{ float: 'right' }}
                    icon={<ReloadOutlined />}
                    onClick={loadDatabaseStats}
                    loading={databaseLoading}
                  >
                    รีเฟรช
                  </Button>
                </Title>
                
                {databaseStats ? (
                  <Row gutter={[16, 16]}>
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small">
                        <Statistic
                          title="ผู้ใช้ทั้งหมด"
                          value={databaseStats.users.total}
                          prefix={<UserOutlined />}
                        />
                        <Text type="secondary">
                          ใช้งาน: {databaseStats.users.active} | 
                          ไม่ใช้งาน: {databaseStats.users.inactive}
                        </Text>
                      </Card>
                    </Col>
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small">
                        <Statistic
                          title="ข้อความทั้งหมด"
                          value={databaseStats.chats.total}
                          prefix={<MessageOutlined />}
                        />
                        <Text type="secondary">
                          วันนี้: {databaseStats.chats.today}
                        </Text>
                      </Card>
                    </Col>
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small">
                        <Statistic
                          title="Embeddings"
                          value={databaseStats.embeddings.total}
                          prefix={<FileImageOutlined />}
                        />
                        <Text type="secondary">
                          ขนาด: {(databaseStats.embeddings.totalSize / 1024).toFixed(2)} KB
                        </Text>
                      </Card>
                    </Col>
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small">
                        <Statistic
                          title="ขนาดฐานข้อมูล"
                          value={databaseStats.system.dbSize}
                          prefix={<DatabaseOutlined />}
                        />
                        <Text type="secondary">
                          อัพไทม์: {databaseStats.system.uptime}
                        </Text>
                      </Card>
                    </Col>
                  </Row>
                ) : (
                  <Button 
                    type="primary" 
                    icon={<ReloadOutlined />}
                    onClick={loadDatabaseStats}
                    loading={databaseLoading}
                  >
                    โหลดสถิติฐานข้อมูล
                  </Button>
                )}
              </Card>

              {/* System Performance */}
              {databaseStats && (
                <Card style={cardStyle}>
                  <Title level={4}>ประสิทธิภาพระบบ</Title>
                  <Row gutter={[16, 16]}>
                    <Col xs={24} md={8}>
                      <Statistic
                        title="เวลาตอบสนองเฉลี่ย"
                        value={databaseStats.system.performance.avgResponseTime}
                        suffix="ms"
                        valueStyle={{ 
                          color: databaseStats.system.performance.avgResponseTime < 100 ? '#3f8600' : '#cf1322' 
                        }}
                      />
                    </Col>
                    <Col xs={24} md={8}>
                      <Statistic
                        title="Query ทั้งหมด"
                        value={databaseStats.system.performance.totalQueries}
                      />
                    </Col>
                    <Col xs={24} md={8}>
                      <Statistic
                        title="Slow Queries"
                        value={databaseStats.system.performance.slowQueries}
                        valueStyle={{ 
                          color: databaseStats.system.performance.slowQueries > 50 ? '#cf1322' : '#3f8600' 
                        }}
                      />
                    </Col>
                  </Row>
                  <Divider />
                  <Text>
                    <strong>การสำรองข้อมูลล่าสุด:</strong> {databaseStats.system.lastBackup}
                  </Text>
                </Card>
              )}

              {/* Database Operations */}
              <Card style={cardStyle}>
                <Title level={4}>จัดการฐานข้อมูล</Title>
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  <Alert
                    message="คำแนะนำ"
                    description="การดำเนินการเหล่านี้จะส่งผลกระทบต่อระบบ กรุณาใช้ความระมัดระวัง"
                    type="warning"
                    showIcon
                  />
                  
                  <Row gutter={[16, 16]}>
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <CloudUploadOutlined style={{ fontSize: '24px', color: colors.light.primary }} />
                        <Title level={5}>สำรองข้อมูล</Title>
                        <Button 
                          type="primary"
                          loading={operationLoading}
                          onClick={() => performDatabaseOperation('backup')}
                        >
                          สำรองข้อมูล
                        </Button>
                      </Card>
                    </Col>
                    
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <ToolOutlined style={{ fontSize: '24px', color: '#fa8c16' }} />
                        <Title level={5}>ปรับปรุงประสิทธิภาพ</Title>
                        <Button 
                          loading={operationLoading}
                          onClick={() => performDatabaseOperation('optimize')}
                        >
                          ปรับปรุง
                        </Button>
                      </Card>
                    </Col>
                    
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <ClearOutlined style={{ fontSize: '24px', color: '#722ed1' }} />
                        <Title level={5}>ล้างข้อมูลเก่า</Title>
                        <Popconfirm
                          title="ล้างข้อความเก่ากว่า 30 วัน?"
                          onConfirm={() => performDatabaseOperation('cleanup', 'chats', { olderThanDays: 30 })}
                        >
                          <Button 
                            loading={operationLoading}
                          >
                            ล้างข้อมูล
                          </Button>
                        </Popconfirm>
                      </Card>
                    </Col>
                    
                    <Col xs={24} sm={12} md={6}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <ToolOutlined style={{ fontSize: '24px', color: '#52c41a' }} />
                        <Title level={5}>ซ่อมแซม</Title>
                        <Button 
                          loading={operationLoading}
                          onClick={() => performDatabaseOperation('repair')}
                        >
                          ซ่อมแซม
                        </Button>
                      </Card>
                    </Col>
                  </Row>
                </Space>
              </Card>

              {/* Embeddings Management */}
              <Card style={cardStyle}>
                <Title level={4}>
                  <FileImageOutlined /> จัดการ Face Embeddings
                  <Space style={{ float: 'right' }}>
                    <Select
                      value={embeddingFilter}
                      onChange={(value) => {
                        setEmbeddingFilter(value);
                        loadEmbeddings();
                      }}
                      style={{ width: 120 }}
                    >
                      <Option value="all">ทั้งหมด</Option>
                      <Option value="active">ใช้งาน</Option>
                      <Option value="inactive">ไม่ใช้งาน</Option>
                    </Select>
                    <Button 
                      icon={<ReloadOutlined />}
                      onClick={loadEmbeddings}
                      loading={embeddingsLoading}
                    >
                      รีเฟรช
                    </Button>
                  </Space>
                </Title>

                {embeddingStats && (
                  <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
                    <Col xs={24} sm={8}>
                      <Statistic
                        title="Embeddings ทั้งหมด"
                        value={embeddingStats.total}
                        prefix={<FileOutlined />}
                      />
                    </Col>
                    <Col xs={24} sm={8}>
                      <Statistic
                        title="ผู้ใช้ที่มี Face Data"
                        value={embeddingStats.uniqueUsers}
                        prefix={<UserOutlined />}
                      />
                    </Col>
                    <Col xs={24} sm={8}>
                      <Statistic
                        title="ความแม่นยำเฉลี่ย"
                        value={embeddingStats.averageAccuracy}
                        suffix="%"
                        precision={1}
                        valueStyle={{ color: embeddingStats.averageAccuracy > 95 ? '#3f8600' : '#fa8c16' }}
                      />
                    </Col>
                  </Row>
                )}

                <Space style={{ marginBottom: '16px' }}>
                  <Button
                    type="primary"
                    icon={<CheckCircleOutlined />}
                    disabled={selectedEmbeddings.length === 0}
                    loading={operationLoading}
                    onClick={() => performEmbeddingOperation('activate', selectedEmbeddings)}
                  >
                    เปิดใช้งาน ({selectedEmbeddings.length})
                  </Button>
                  <Button
                    icon={<ExclamationCircleOutlined />}
                    disabled={selectedEmbeddings.length === 0}
                    loading={operationLoading}
                    onClick={() => performEmbeddingOperation('deactivate', selectedEmbeddings)}
                  >
                    ปิดใช้งาน
                  </Button>
                  <Button
                    icon={<ReloadOutlined />}
                    disabled={selectedEmbeddings.length === 0}
                    loading={operationLoading}
                    onClick={() => performEmbeddingOperation('regenerate', selectedEmbeddings)}
                  >
                    สร้างใหม่
                  </Button>
                  <Popconfirm
                    title="ลบ embeddings ที่เลือก?"
                    onConfirm={() => performEmbeddingOperation('delete', selectedEmbeddings)}
                  >
                    <Button
                      danger
                      icon={<DeleteOutlined />}
                      disabled={selectedEmbeddings.length === 0}
                      loading={operationLoading}
                    >
                      ลบ
                    </Button>
                  </Popconfirm>
                  <Button
                    icon={<ClearOutlined />}
                    loading={operationLoading}
                    onClick={() => performEmbeddingOperation('cleanup_orphaned')}
                  >
                    ล้างไฟล์เก่า
                  </Button>
                  <Button
                    icon={<ScanOutlined />}
                    loading={operationLoading}
                    onClick={() => performEmbeddingOperation('analyze_duplicates')}
                  >
                    วิเคราะห์ซ้ำ
                  </Button>
                </Space>

                {!embeddingsLoading && embeddings.length === 0 ? (
                  <Button 
                    type="primary" 
                    icon={<ReloadOutlined />}
                    onClick={loadEmbeddings}
                  >
                    โหลดข้อมูล Embeddings
                  </Button>
                ) : (                  <Table
                    rowSelection={{
                      selectedRowKeys: selectedEmbeddings,
                      onChange: (selectedRowKeys) => setSelectedEmbeddings(selectedRowKeys as string[]),
                    }}
                    dataSource={embeddings}
                    loading={embeddingsLoading}
                    rowKey="id"
                    pagination={{
                      pageSize: 10,
                      showSizeChanger: true,
                      showQuickJumper: true,
                      showTotal: (total, range) => `${range[0]}-${range[1]} จาก ${total} รายการ`
                    }}
                    columns={[
                      {
                        title: 'ไฟล์',
                        key: 'file',
                        render: (record: EmbeddingFile) => (
                          <Space>
                            <FileImageOutlined style={{ fontSize: '16px' }} />
                            <div>
                              <Text strong>{record.fileName}</Text>
                              <br />
                              <Text type="secondary">{record.format.toUpperCase()} • {(record.size / 1024).toFixed(1)} KB</Text>
                            </div>
                          </Space>
                        )
                      },
                      {
                        title: 'ผู้ใช้',
                        dataIndex: 'userId',
                        key: 'userId',
                        render: (userId: string) => (
                          <Tag color="blue">{userId}</Tag>
                        )
                      },
                      {
                        title: 'ความแม่นยำ',
                        dataIndex: 'accuracy',
                        key: 'accuracy',
                        render: (accuracy: number) => (
                          <Progress 
                            percent={accuracy} 
                            size="small" 
                            status={accuracy > 95 ? 'success' : accuracy > 90 ? 'normal' : 'exception'}
                          />
                        )
                      },
                      {
                        title: 'สถานะ',
                        dataIndex: 'isActive',
                        key: 'isActive',
                        render: (isActive: boolean) => (
                          <Badge 
                            status={isActive ? 'success' : 'error'} 
                            text={isActive ? 'ใช้งาน' : 'ไม่ใช้งาน'} 
                          />
                        )
                      },
                      {
                        title: 'วันที่สร้าง',
                        dataIndex: 'createdAt',
                        key: 'createdAt',
                        render: (date: string) => new Date(date).toLocaleDateString('th-TH')
                      },
                      {
                        title: 'ใช้ล่าสุด',
                        dataIndex: 'lastUsed',
                        key: 'lastUsed',
                        render: (date: string) => new Date(date).toLocaleDateString('th-TH')
                      }
                    ]}
                  />
                )}
              </Card>
            </Space>
          </TabPane>

          {/* System Logs Tab */}
          <TabPane tab={<span><FileOutlined />จัดการ System Logs</span>} key="logs">
            <Card style={cardStyle}>
              <Title level={4}>
                <FileOutlined /> System Logs
                <Button 
                  style={{ float: 'right' }}
                  icon={<ReloadOutlined />}
                  onClick={loadLogs}
                  loading={logsLoading}
                >
                  รีเฟรช
                </Button>
              </Title>
              
              <Table
                columns={[
                  {
                    title: 'ระดับ',
                    dataIndex: 'level',
                    key: 'level',
                    render: (level: string) => {
                      let color = 'default';
                      if (level === 'error') color = 'red';
                      else if (level === 'warn') color = 'orange';
                      else if (level === 'info') color = 'blue';
                      else if (level === 'debug') color = 'green';
                      
                      return (
                        <Tag color={color} style={{ borderRadius: '4px' }}>
                          {level.toUpperCase()}
                        </Tag>
                      );
                    }
                  },
                  {
                    title: 'แหล่งที่มา',
                    dataIndex: 'source',
                    key: 'source'
                  },
                  {
                    title: 'ข้อความ',
                    dataIndex: 'message',
                    key: 'message',
                    render: (text: string) => (
                      <Tooltip title={text}>
                        <Text ellipsis>{text}</Text>
                      </Tooltip>
                    )
                  },
                  {
                    title: 'วันที่เวลา',
                    dataIndex: 'timestamp',
                    key: 'timestamp',
                    render: (timestamp: string) => new Date(timestamp).toLocaleString('th-TH')
                  }
                ]}
                dataSource={logs}
                rowKey="id"
                loading={logsLoading}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `ทั้งหมด ${total} รายการ`
                }}
              />
            </Card>
          </TabPane>
        </Tabs>

        {/* Edit User Modal */}
        <Modal
          title="แก้ไขข้อมูลผู้ใช้"
          open={editModalVisible}
          onOk={handleSaveUser}
          onCancel={() => setEditModalVisible(false)}
          okText="บันทึก"
          cancelText="ยกเลิก"
        >
          <Form form={form} layout="vertical">
            <Form.Item
              label="ชื่อจริง"
              name="firstName"
              rules={[{ required: true, message: 'กรุณาใส่ชื่อจริง' }]}
            >
              <Input />
            </Form.Item>
            
            <Form.Item
              label="นามสกุล"
              name="lastName"
              rules={[{ required: true, message: 'กรุณาใส่นามสกุล' }]}
            >
              <Input />
            </Form.Item>
            
            <Form.Item
              label="ชื่อผู้ใช้"
              name="username"
              rules={[{ required: true, message: 'กรุณาใส่ชื่อผู้ใช้' }]}
            >
              <Input />
            </Form.Item>
            
            <Form.Item
              label="อีเมล"
              name="email"
              rules={[
                { required: true, message: 'กรุณาใส่อีเมล' },
                { type: 'email', message: 'รูปแบบอีเมลไม่ถูกต้อง' }
              ]}
            >
              <Input />
            </Form.Item>
            
            <Form.Item
              label="เบอร์โทร"
              name="phone"
            >
              <Input />
            </Form.Item>
            
            <Form.Item
              label="สถานะการใช้งาน"
              name="isActive"
              valuePropName="checked"
            >
              <Select>
                <Option value={true}>ใช้งาน</Option>
                <Option value={false}>ระงับ</Option>
              </Select>
            </Form.Item>
            
            <Form.Item
              label="สถานะการยืนยัน"
              name="isVerified"
              valuePropName="checked"
            >
              <Select>
                <Option value={true}>ยืนยันแล้ว</Option>
                <Option value={false}>รอยืนยัน</Option>
              </Select>
            </Form.Item>
          </Form>
        </Modal>
      </div>
    </AppLayout>
  );
};

export default AdminDashboard;
