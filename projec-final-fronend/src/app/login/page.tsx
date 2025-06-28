'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { 
  Card, 
  Form, 
  Input, 
  Button, 
  Typography, 
  Alert,
  Divider,
  message
} from 'antd';
import { 
  LockOutlined,
  MailOutlined,
  LoginOutlined,
  ScanOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import FaceLoginModal from '../../components/auth/FaceLoginModal';
import FaceAPIStatus from '@/components/ai/FaceAPIStatus';

const { Title, Text, Paragraph } = Typography;

interface LoginForm {
  identifier: string;  // Can be email or username
  password: string;
}

const LoginPage: React.FC = () => {
  const router = useRouter();
  const [form] = Form.useForm();  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvancedFaceScan, setShowAdvancedFaceScan] = useState(false);

  const handleLogin = async (values: LoginForm) => {
    try {
      setLoading(true);
      setError(null);

      // Determine if the identifier is an email or username
      const isEmail = values.identifier.includes('@');
      const loginData = {
        [isEmail ? 'email' : 'username']: values.identifier,
        password: values.password,
        loginMethod: 'password'
      };

      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(loginData)
      });

      const result = await response.json();

      if (response.ok && result.success) {
        message.success('เข้าสู่ระบบสำเร็จ!');
        
        // Store user data in localStorage (in production, use proper state management)
        localStorage.setItem('user', JSON.stringify(result.data.user));
        localStorage.setItem('token', result.data.token);
        
        // Redirect to dashboard
        router.push('/dashboard');
      } else {
        setError(result.message || 'เกิดข้อผิดพลาดในการเข้าสู่ระบบ');
      }    } catch (error) {
      console.error('Login error:', error);
      setError('เกิดข้อผิดพลาดในการเชื่อมต่อ กรุณาลองใหม่อีกครั้ง');
    } finally {
      setLoading(false);
    }
  };  const handleFaceLoginSuccess = async (result: any) => {
    console.log('🎉 Face login success result:', result);
    
    // ปิด modal
    setShowAdvancedFaceScan(false);
    
    // ตรวจสอบว่าได้ข้อมูลผู้ใช้จริงจาก API หรือไม่
    if (result.api_result && result.api_result.data && result.api_result.data.user) {
      // ใช้ข้อมูลจาก login API (เหมือน HTML demo)
      const userData = result.api_result.data.user;
      const token = result.api_result.data.token;
      
      console.log('✅ Using data from login API:', userData);
      message.success(`ยินดีต้อนรับ ${userData.fullName || userData.username}!`);
      
      // Store user data เหมือน password login
      localStorage.setItem('user', JSON.stringify(userData));
      localStorage.setItem('token', token);
      
      // Redirect to dashboard
      router.push('/dashboard');    } else if (result.user && result.user.identity !== undefined) {
      // Fallback ถ้าไม่ได้ข้อมูลจาก login API
      console.warn('⚠️ ไม่ได้ข้อมูลจาก login API - ใช้ข้อมูลจากการสแกนใบหน้า');
      console.log('📋 Face scan user data:', result.user);
      message.success(`เข้าสู่ระบบสำเร็จ - ตรวจพบใบหน้า: ${result.user.fullName || result.user.username}`);
      
      // ใช้ข้อมูลที่ดึงมาจาก Face Recognition API แล้ว
      const userData = {
        id: result.user.id,
        identity: result.user.identity,
        username: result.user.username,
        email: result.user.email,
        firstName: result.user.firstName,
        lastName: result.user.lastName,
        fullName: result.user.fullName,
        isVerified: result.user.isVerified || false
      };
      console.log('💾 Storing user data:', userData);
      localStorage.setItem('user', JSON.stringify(userData));
      
      // สร้าง JWT token ที่ถูกต้องสำหรับ Face Login
      if (result.token && result.token.includes('.')) {
        // ถ้ามี JWT token ที่ถูกต้อง
        localStorage.setItem('token', result.token);
        setTimeout(() => {
          router.push('/dashboard');
        }, 100);
      } else {
        // หากไม่มี token หรือ token ไม่ถูกต้อง ให้เรียก login API
        console.log('🔄 Creating proper JWT token for face login...');
        try {
          const loginResponse = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              loginMethod: 'face',
              userId: result.user.id,
              username: result.user.username
            })
          });
          
          if (loginResponse.ok) {
            const loginResult = await loginResponse.json();
            console.log('🔑 Login API response:', loginResult);
            
            if (loginResult.success && loginResult.data?.token) {
              localStorage.setItem('token', loginResult.data.token);
              console.log('✅ JWT token created and stored successfully');
              
              // เพิ่ม delay เล็กน้อยเพื่อให้ localStorage update
              setTimeout(() => {
                router.push('/dashboard');
              }, 100);
            } else {
              console.error('Login API response missing token:', loginResult);
              message.error('ไม่สามารถสร้าง token ได้');
              return;
            }
          } else {
            console.error('Failed to create JWT token');
            message.error('เกิดข้อผิดพลาดในการสร้าง token');
            return;
          }
        } catch (error) {
          console.error('Error creating JWT token:', error);
          message.error('เกิดข้อผิดพลาดในการสร้าง token');
          return;
        }
      }
      
    } else {
      message.error('ไม่สามารถดึงข้อมูลผู้ใช้ได้');
    }
  };
  const handleFaceLoginError = (error: string) => {
    setError(error);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-blue-600">🤖 FaceSocial</h1>
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <Link href="/" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">
                  หน้าแรก
                </Link>
                <Link href="/register" className="text-blue-600 hover:text-blue-800 px-3 py-2 rounded-md text-sm font-medium">
                  สมัครสมาชิก
                </Link>
              </div>
            </div>          </div>
        </div>
      </nav>
      {/* Main Content */}
      <div className="py-12 px-4">
        <div className="max-w-md mx-auto">
          {/* Face API Status */}
          <div className="mb-6">
            <FaceAPIStatus />
          </div>
          
          <Card className="shadow-lg">
            <div className="text-center mb-8">
              <LoginOutlined className="text-5xl text-blue-500 mb-4" />
              <Title level={2}>เข้าสู่ระบบ</Title>
              <Paragraph type="secondary">
                เข้าสู่ระบบด้วยอีเมลและรหัสผ่าน หรือใบหน้า
              </Paragraph>
            </div>

            {/* Error Alert */}
            {error && (
              <Alert
                message="เกิดข้อผิดพลาด"
                description={error}
                type="error"
                showIcon
                closable
                onClose={() => setError(null)}
                className="mb-6"
              />
            )}

            <Form
              form={form}
              layout="vertical"
              onFinish={handleLogin}
              autoComplete="off"
              size="large"
            >              <Form.Item
                name="identifier"
                label="อีเมลหรือชื่อผู้ใช้"
                rules={[
                  { required: true, message: 'กรุณากรอกอีเมลหรือชื่อผู้ใช้' }
                ]}
              >
                <Input 
                  prefix={<MailOutlined />} 
                  placeholder="อีเมลหรือชื่อผู้ใช้" 
                />
              </Form.Item>

              <Form.Item
                name="password"
                label="รหัสผ่าน"
                rules={[
                  { required: true, message: 'กรุณากรอกรหัสผ่าน' }
                ]}
              >
                <Input.Password 
                  prefix={<LockOutlined />} 
                  placeholder="รหัสผ่าน" 
                />
              </Form.Item>              <Form.Item>
                <Button 
                  type="primary" 
                  htmlType="submit" 
                  loading={loading}
                  size="large"
                  block
                  icon={<LoginOutlined />}
                >
                  เข้าสู่ระบบด้วยรหัสผ่าน
                </Button>
              </Form.Item>
            </Form>

            <Divider>หรือ</Divider>

            {/* Face Scan Login Button */}
            <div className="mb-6">              <Button
                type="default"
                size="large"
                block
                icon={<ScanOutlined />}
                onClick={() => setShowAdvancedFaceScan(true)}
                className="border-2 border-blue-500 text-blue-500 hover:bg-blue-50"
              >
                เข้าสู่ระบบด้วยใบหน้า (Advanced)
              </Button>
            </div>

            <div className="text-center space-y-4">
              <div>
                <Text type="secondary">
                  ยังไม่มีบัญชี? <a href="/register" className="text-blue-600 hover:text-blue-800">สมัครสมาชิก</a>
                </Text>
              </div>
              
              <div>
                <a href="/forgot-password" className="text-blue-600 hover:text-blue-800 text-sm">
                  ลืมรหัสผ่าน?
                </a>
              </div>
            </div>            {/* Demo Account Info */}
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <Text strong className="text-blue-800">บัญชีทดสอบ:</Text>
              <div className="mt-2 text-sm text-blue-700">
                <div>อีเมล: admin@email.com</div>
                <div>ชื่อผู้ใช้: admin01</div>
                <div>รหัสผ่าน: admin01</div>
                <div className="mt-1 text-xs text-green-600">✅ รองรับการเข้าสู่ระบบด้วยใบหน้า</div>
              </div>
            </div>
          </Card>        </div>
      </div>
      {/* Face Login Modal */}
      <FaceLoginModal
        visible={showAdvancedFaceScan}
        onClose={() => setShowAdvancedFaceScan(false)}
        onSuccess={handleFaceLoginSuccess}
        onError={handleFaceLoginError}
      />
    </div>
  );
};

export default LoginPage;
