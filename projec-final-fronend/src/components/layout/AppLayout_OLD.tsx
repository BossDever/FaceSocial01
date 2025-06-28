'use client';

import React, { useState, useEffect } from 'react';
import { 
  Layout, 
  Menu, 
  Avatar, 
  Dropdown, 
  Button, 
  Space, 
  Badge,
  Typography,
  message,
  Tooltip
} from 'antd';
import { 
  DashboardOutlined,
  AppstoreOutlined,
  MessageOutlined,
  UserOutlined,
  LogoutOutlined,
  BellOutlined,
  SettingOutlined,
  UserAddOutlined,
  PlusOutlined,
  TeamOutlined,
  MenuUnfoldOutlined,
  MenuFoldOutlined
} from '@ant-design/icons';
import { useRouter, usePathname } from 'next/navigation';
import type { MenuProps } from 'antd';
import NotificationDropdown from '../social/NotificationDropdown';
import FriendSearchModal from '../social/FriendSearchModal';
import PostCreateModal from '../social/PostCreateModal';
import { colors, layoutStyle, menuStyle, buttonStyle, iconStyle, textStyle } from '../../styles/theme';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;

interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  profilePicture?: string;
}

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [friendSearchVisible, setFriendSearchVisible] = useState(false);
  const [postCreateVisible, setPostCreateVisible] = useState(false);

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
      } else {
        router.push('/login');
      }
    } catch (error) {
      console.error('Error fetching user:', error);
      router.push('/login');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    message.success('ออกจากระบบสำเร็จ');
    router.push('/login');
  };  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined style={iconStyle.primary} />,
      label: (
        <span style={{ color: colors.light.textPrimary, fontWeight: 500 }}>
          แดชบอร์ด
        </span>
      ),
      onClick: () => router.push('/dashboard')
    },
    {
      key: '/feed',
      icon: <AppstoreOutlined style={iconStyle.primary} />,
      label: (
        <span style={{ color: colors.light.textPrimary, fontWeight: 500 }}>
          ฟีด
        </span>
      ),
      onClick: () => router.push('/feed')
    },
    {
      key: '/friends',
      icon: <TeamOutlined style={iconStyle.primary} />,
      label: (
        <span style={{ color: colors.light.textPrimary, fontWeight: 500 }}>
          เพื่อน
        </span>
      ),
      onClick: () => router.push('/friends')
    },
    {
      key: '/chat',
      icon: <MessageOutlined style={iconStyle.primary} />,
      label: (
        <span style={{ color: colors.light.textPrimary, fontWeight: 500 }}>
          แชท
        </span>
      ),
      onClick: () => router.push('/chat')
    },
    {
      key: '/notifications',
      icon: <BellOutlined style={iconStyle.primary} />,
      label: (
        <span style={{ color: colors.light.textPrimary, fontWeight: 500 }}>
          การแจ้งเตือน
        </span>
      ),
      onClick: () => router.push('/notifications')
    },
    {
      key: '/profile',
      icon: <UserOutlined style={iconStyle.primary} />,
      label: (
        <span style={{ color: colors.light.textPrimary, fontWeight: 500 }}>
          โปรไฟล์
        </span>
      ),
      onClick: () => router.push('/profile')
    }
  ];
  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined style={iconStyle.secondary} />,
      label: (
        <span style={{ color: colors.light.textPrimary }}>โปรไฟล์</span>
      ),
      onClick: () => router.push('/profile')
    },
    {
      key: 'settings',
      icon: <SettingOutlined style={iconStyle.secondary} />,
      label: (
        <span style={{ color: colors.light.textPrimary }}>ตั้งค่า</span>
      ),
      onClick: () => message.info('ฟีเจอร์นี้กำลังพัฒนา')
    },
    {
      type: 'divider'
    },
    {
      key: 'logout',
      icon: <LogoutOutlined style={iconStyle.error} />,
      label: (
        <span style={{ color: colors.light.error }}>ออกจากระบบ</span>
      ),
      onClick: handleLogout,
      danger: true
    }
  ];
  if (!user) {
    return (
      <div 
        style={{ 
          minHeight: '100vh',
          backgroundColor: colors.light.bgSecondary,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div 
            style={{
              width: '48px',
              height: '48px',
              border: `3px solid ${colors.light.borderPrimary}`,
              borderTop: `3px solid ${colors.light.primary}`,
              borderRadius: '50%',
              margin: '0 auto 16px',
              animation: 'spin 1s linear infinite'
            }}
          />
          <Text style={{ color: colors.light.textSecondary, fontSize: '16px' }}>
            กำลังโหลด...
          </Text>
        </div>
      </div>
    );
  }
  return (
    <Layout style={{ minHeight: '100vh', backgroundColor: colors.light.bgSecondary }}>
      {/* Sidebar */}
      <Sider 
        collapsible 
        collapsed={collapsed} 
        onCollapse={setCollapsed}
        width={260}
        style={{
          ...layoutStyle.sider,
          position: 'fixed',
          height: '100vh',
          left: 0,
          zIndex: 100,
        }}
        trigger={null}
      >
        {/* Logo */}
        <div 
          style={{
            padding: collapsed ? '20px 12px' : '20px 24px',
            textAlign: 'center',
            borderBottom: `1px solid ${colors.light.borderSecondary}`,
            backgroundColor: colors.light.bgContainer,
          }}
        >
          {!collapsed ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ fontSize: '24px', marginRight: '8px' }}>🤖</span>
              <Title 
                level={3} 
                style={{ 
                  ...textStyle.heading3,
                  margin: 0,
                  background: colors.brand.gradient,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                FaceSocial
              </Title>
            </div>
          ) : (
            <div style={{ fontSize: '28px' }}>🤖</div>
          )}
        </div>

        {/* Navigation Menu */}
        <Menu
          mode="inline"
          selectedKeys={pathname ? [pathname] : []}
          items={menuItems}
          style={{
            backgroundColor: 'transparent',
            border: 'none',
            padding: '16px 8px',
            fontSize: '14px',
          }}
          className="custom-menu"
        />

        {/* User Info in Sidebar (when expanded) */}
        {!collapsed && (
          <div 
            style={{
              position: 'absolute',
              bottom: '16px',
              left: '16px',
              right: '16px',
              padding: '16px',
              backgroundColor: `${colors.light.primary}08`,
              borderRadius: '12px',
              border: `1px solid ${colors.light.borderSecondary}`,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <Avatar 
                src={user.profilePicture} 
                icon={<UserOutlined />}
                size={40}
                style={{ marginRight: '12px' }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ 
                  ...textStyle.body,
                  fontWeight: 600,
                  marginBottom: '2px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>
                  {user.firstName} {user.lastName}
                </div>
                <div style={{ 
                  ...textStyle.caption,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>
                  @{user.username}
                </div>
              </div>
            </div>
          </div>
        )}
      </Sider>

      <Layout style={{ marginLeft: collapsed ? 80 : 260, transition: 'all 0.2s' }}>
        {/* Header */}
        <Header 
          style={{
            ...layoutStyle.header,
            height: '64px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'sticky',
            top: 0,
            zIndex: 10,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{
                fontSize: '16px',
                width: 48,
                height: 48,
                marginRight: '16px',
                color: colors.light.textSecondary,
              }}
            />
            <Title 
              level={4} 
              style={{ 
                ...textStyle.heading3,
                margin: 0,
              }}
            >
              {pathname === '/dashboard' && 'แดชบอร์ด'}
              {pathname === '/feed' && 'ฟีด'}
              {pathname === '/chat' && 'แชท'}
              {pathname === '/friends' && 'เพื่อน'}
              {pathname === '/profile' && 'โปรไฟล์'}
              {pathname === '/notifications' && 'การแจ้งเตือน'}
            </Title>
          </div>

          <Space size={16}>
            {/* Create Post Button */}
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => setPostCreateVisible(true)}
              style={{
                ...buttonStyle.primary,
                display: window.innerWidth > 768 ? 'flex' : 'none',
                alignItems: 'center',
              }}
            >
              สร้างโพสต์
            </Button>

            {/* Find Friends Button */}
            <Tooltip title="หาเพื่อน">
              <Button 
                icon={<UserAddOutlined style={iconStyle.secondary} />}
                onClick={() => setFriendSearchVisible(true)}
                style={buttonStyle.ghost}
              />
            </Tooltip>
            >
              <span className="hidden md:inline ml-1">หาเพื่อน</span>
            </Button>

            {/* Notifications */}
            <NotificationDropdown />

            {/* User Dropdown */}
            <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
              <Button type="text" className="flex items-center space-x-2 px-2">
                <Avatar 
                  src={user.profilePicture} 
                  icon={<UserOutlined />}
                  size="small"
                />
                <div className="hidden md:block text-left">
                  <Text className="text-gray-800 text-sm font-medium block">
                    {user.firstName} {user.lastName}
                  </Text>
                  <Text className="text-gray-500 text-xs">
                    @{user.username}
                  </Text>
                </div>
              </Button>
            </Dropdown>
          </Space>
        </Header>        {/* Main Content */}
        <Content className="p-6 bg-gray-50">
          {children}
        </Content>
      </Layout>

      {/* Modals */}
      <FriendSearchModal 
        visible={friendSearchVisible}
        onCancel={() => setFriendSearchVisible(false)}
      />
      
      <PostCreateModal 
        visible={postCreateVisible}
        onCancel={() => setPostCreateVisible(false)}
        onSuccess={(post) => {
          setPostCreateVisible(false);
          message.success('โพสต์สำเร็จ!');
          // รีเฟรชหน้าฟีดถ้าอยู่ในหน้าฟีด
          if (pathname === '/feed') {
            window.location.reload();
          }
        }}
      />
    </Layout>
  );
};

export default AppLayout;
