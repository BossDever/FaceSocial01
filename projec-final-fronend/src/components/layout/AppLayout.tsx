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
  };

  // Determine menu items based on user role
  const getMenuItems = () => {
    const baseItems = [
      {
        key: '/dashboard',
        icon: <DashboardOutlined style={iconStyle.primary} />,
        label: 'แดชบอร์ด',
        onClick: () => router.push('/dashboard')
      },
      {
        key: '/feed',
        icon: <AppstoreOutlined style={iconStyle.primary} />,
        label: 'ฟีด',
        onClick: () => router.push('/feed')
      },
      {
        key: '/friends',
        icon: <TeamOutlined style={iconStyle.primary} />,
        label: 'เพื่อน',
        onClick: () => router.push('/friends')
      },
      {
        key: '/chat',
        icon: <MessageOutlined style={iconStyle.primary} />,
        label: 'แชท',
        onClick: () => router.push('/chat')
      },
      {
        key: '/notifications',
        icon: <BellOutlined style={iconStyle.primary} />,
        label: 'การแจ้งเตือน',
        onClick: () => router.push('/notifications')
      },
      {
        key: '/profile',
        icon: <UserOutlined style={iconStyle.primary} />,
        label: 'โปรไฟล์',
        onClick: () => router.push('/profile')
      }
    ];

    // Add admin menu for admin users
    if (user?.username === 'admin01') {
      baseItems.push({
        key: '/admin',
        icon: <SettingOutlined style={iconStyle.primary} />,
        label: 'จัดการระบบ',
        onClick: () => router.push('/admin')
      });
    }

    return baseItems;
  };

  const menuItems = getMenuItems();

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined style={iconStyle.secondary} />,
      label: 'โปรไฟล์',
      onClick: () => router.push('/profile')
    },
    {
      key: 'settings',
      icon: <SettingOutlined style={iconStyle.secondary} />,
      label: 'ตั้งค่า',
      onClick: () => message.info('ฟีเจอร์นี้กำลังพัฒนา')
    },
    {
      type: 'divider'
    },
    {
      key: 'logout',
      icon: <LogoutOutlined style={iconStyle.error} />,
      label: 'ออกจากระบบ',
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

  const getPageTitle = () => {
    switch(pathname) {
      case '/dashboard': return 'แดชบอร์ด';
      case '/feed': return 'ฟีด';
      case '/chat': return 'แชท';
      case '/friends': return 'เพื่อน';
      case '/profile': return 'โปรไฟล์';
      case '/notifications': return 'การแจ้งเตือน';
      default: return 'FaceSocial';
    }
  };

  return (
    <>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .custom-menu .ant-menu-item {
          color: ${colors.light.textSecondary} !important;
          margin: 4px 0 !important;
          border-radius: 8px !important;
          height: 48px !important;
          line-height: 48px !important;
          transition: all 0.3s ease !important;
        }
        
        .custom-menu .ant-menu-item:hover {
          background-color: ${colors.light.bgHover} !important;
          color: ${colors.light.primary} !important;
        }
        
        .custom-menu .ant-menu-item-selected {
          background-color: ${colors.light.primary}15 !important;
          color: ${colors.light.primary} !important;
          font-weight: 600 !important;
        }
        
        .custom-menu .ant-menu-item-icon {
          font-size: 18px !important;
          margin-right: 12px !important;
        }
        
        .ant-layout-sider-trigger {
          background-color: ${colors.light.bgContainer} !important;
          color: ${colors.light.textPrimary} !important;
          border-top: 1px solid ${colors.light.borderSecondary} !important;
        }
      `}</style>
      
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
          />          {/* User Info in Sidebar (when expanded) */}
          {!collapsed && (
            <div 
              style={{
                position: 'absolute',
                bottom: '120px', // เพิ่มระยะห่างจากด้านล่างมากขึ้น
                left: '16px',
                right: '16px',
                padding: '16px',
                backgroundColor: `${colors.light.primary}08`,
                borderRadius: '12px',
                border: `1px solid ${colors.light.borderSecondary}`,
                zIndex: 1000, // เพิ่ม z-index สูงขึ้น
                maxHeight: '80px', // จำกัดความสูง
                overflow: 'hidden', // ป้องกันการล้น
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <Avatar 
                  src={user.profilePicture} 
                  icon={<UserOutlined />}
                  size={40}
                  style={{ marginRight: '12px', flexShrink: 0 }}
                />
                <div style={{ 
                  flex: 1, 
                  minWidth: 0, 
                  overflow: 'hidden',
                  width: '100%'
                }}>
                  <div style={{ 
                    ...textStyle.body,
                    fontWeight: 600,
                    marginBottom: '2px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    width: '100%',
                    maxWidth: '150px' // จำกัดความกว้าง
                  }}>
                    {user.firstName} {user.lastName}
                  </div>
                  <div style={{ 
                    ...textStyle.caption,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    width: '100%',
                    maxWidth: '150px' // จำกัดความกว้าง
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
              <Title 
                level={4} 
                style={{ 
                  ...textStyle.heading3,
                  margin: 0,
                }}
              >
                {getPageTitle()}
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
                  display: 'flex',
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

              {/* Notifications */}
              <NotificationDropdown />

              {/* User Menu */}
              <Dropdown 
                menu={{ items: userMenuItems }}
                placement="bottomRight"
                trigger={['click']}
              >
                <Button
                  style={{
                    ...buttonStyle.ghost,
                    padding: '4px 8px',
                    height: 'auto',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  <Avatar 
                    src={user.profilePicture} 
                    icon={<UserOutlined />}
                    size={32}
                    style={{ marginRight: '8px' }}
                  />
                  <div style={{ textAlign: 'left', display: 'flex', flexDirection: 'column' }}>
                    <Text style={{ ...textStyle.body, fontWeight: 600, lineHeight: 1.2 }}>
                      {user.firstName}
                    </Text>
                    <Text style={{ ...textStyle.caption, lineHeight: 1.2 }}>
                      @{user.username}
                    </Text>
                  </div>
                </Button>
              </Dropdown>
            </Space>
          </Header>

          {/* Main Content */}
          <Content
            style={{
              ...layoutStyle.content,
              overflow: 'auto',
            }}
          >
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
          onSuccess={() => {
            setPostCreateVisible(false);
            message.success('สร้างโพสต์สำเร็จ');
          }}
        />
      </Layout>
    </>
  );
};

export default AppLayout;
