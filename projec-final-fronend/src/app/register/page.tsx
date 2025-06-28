'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import Link from 'next/link';
import { 
  Card, 
  Steps, 
  Form, 
  Input, 
  Button, 
  Row, 
  Col, 
  Typography, 
  Space,
  Alert,
  DatePicker,
  Checkbox,
  Spin
} from 'antd';
import { 
  UserOutlined, 
  MailOutlined, 
  LockOutlined,
  PhoneOutlined,
  CameraOutlined,
  CheckCircleOutlined,
  ArrowLeftOutlined,
  ArrowRightOutlined,
  LoadingOutlined,
  CheckOutlined,
  CloseOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import FaceRegistration from '@/components/auth/FaceRegistration';
import FaceAPIStatus from '@/components/ai/FaceAPIStatus';
import { useAvailabilityCheck } from '@/hooks/useAvailabilityCheck';

const { Title, Text, Paragraph } = Typography;

interface PersonalInfo {
  firstName: string;
  lastName: string;
  email: string;
  username: string;
  password: string;
  confirmPassword: string;
  phone?: string;
  dateOfBirth?: string;
  agreeToTerms: boolean;
}

interface FaceData {
  embedding: number[];
  imageUrl: string;
  qualityScore: number;
  confidence: number;
  totalImages?: number;
  userInfo?: {
    firstName: string;
    lastName: string;
    userId: string;
  };
}

const RegisterPage: React.FC = () => {
  const router = useRouter();
  const [form] = Form.useForm();
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [personalInfo, setPersonalInfo] = useState<PersonalInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [registrationTriggered, setRegistrationTriggered] = useState(false);

  // ป้องกันการเรียกซ้ำด้วย ref แทน state
  const registrationRef = useRef(false);  // Real-time availability checks
  const emailCheck = useAvailabilityCheck('email');
  const usernameCheck = useAvailabilityCheck('username');
  const phoneCheck = useAvailabilityCheck('phone');
    // Mock nameCheck object เพื่อไม่ให้เกิด error
  const nameCheck = {
    loading: false,
    available: true,
    error: undefined as string | undefined,
    message: undefined as string | undefined,
    check: () => {}
  };

  // Form field values for real-time checking
  const [formValues, setFormValues] = useState({
    firstName: '',
    lastName: '',
    email: '',
    username: '',
    phone: ''
  });
  // Check if all required fields are available
  const isFormValid = () => {
    // Check if we have actual field values
    const hasRequiredValues = formValues.firstName && formValues.lastName && 
                             formValues.email && formValues.username;
    
    if (!hasRequiredValues) return false;
    
    // If we have values, check availability
    const emailValid = !formValues.email || emailCheck.available !== false;
    const usernameValid = !formValues.username || usernameCheck.available !== false;
    const phoneValid = !formValues.phone || phoneCheck.available !== false;
    
    // No loading states
    const noLoadingStates = !emailCheck.loading && !usernameCheck.loading && 
                           !phoneCheck.loading;
    
    return emailValid && usernameValid && phoneValid && noLoadingStates;
  };

  // Step 1: Personal Information
  const handlePersonalInfoSubmit = useCallback(async (values: PersonalInfo) => {
    try {
      setLoading(true);
      setError(null);

      // Validate password match
      if (values.password !== values.confirmPassword) {
        setError('รหัสผ่านไม่ตรงกัน');
        return;
      }

      // Check availability one more time before proceeding
      if (!isFormValid()) {
        setError('กรุณาตรวจสอบข้อมูลที่ป้อนให้ถูกต้อง');
        return;
      }

      setPersonalInfo(values);
      setCurrentStep(1);
      
      console.log('ข้อมูลส่วนตัวถูกต้อง กรุณาสแกนใบหน้าต่อไป');    
    } catch (error) {
      console.error('Form validation error:', error);
      setError('เกิดข้อผิดพลาดในการตรวจสอบข้อมูล');
    } finally {
      setLoading(false);
    }
  }, [isFormValid]);

  // Step 2: Face Registration Success - แก้ไขให้จัดการทุกอย่างใน page.tsx
  const handleFaceRegistrationComplete = useCallback(async (faceData: FaceData) => {
    if (registrationRef.current) {
      console.warn('Registration already in progress, ignoring duplicate call.');
      return;
    }
    
    registrationRef.current = true;
    console.log('🔒 Starting registration process...');
    
    try {
      setLoading(true);
      setError(null);
      
      if (!personalInfo) {
        setError('ไม่พบข้อมูลส่วนตัว กรุณาเริ่มต้นใหม่');
        return;
      }

      const registrationId = `reg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      console.log(`🚀 Starting registration with ID: ${registrationId}`);
      
      const registrationData = {
        ...personalInfo,
        faceEmbedding: faceData.embedding,
        faceImageUrl: faceData.imageUrl,
        faceImageBase64: faceData.imageUrl,
        qualityScore: faceData.qualityScore,
        detectionConfidence: faceData.confidence,
        registrationId
      };

      // เพิ่ม timeout และ retry logic
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(registrationData),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      const result = await response.json();

      if (response.ok && result.success) {
        console.log('✅ Registration successful!');
        setCurrentStep(2);
        
        // Redirect after delay
        setTimeout(() => {
          router.push('/login');
        }, 3000);
          } else {
        console.error('❌ Registration failed:', result);
        
        // Better error handling for different response types
        let errorMessage = 'เกิดข้อผิดพลาดในการสมัครสมาชิก';
        
        if (response.status === 409) {
          errorMessage = 'ผู้ใช้นี้ถูกลงทะเบียนแล้ว กรุณาใช้ข้อมูลอื่น';
        } else if (result?.message) {
          errorMessage = result.message;
        } else if (typeof result === 'string') {
          errorMessage = result;
        } else if (result?.error) {
          errorMessage = typeof result.error === 'string' ? result.error : 'เกิดข้อผิดพลาดของระบบ';
        }
        
        setError(errorMessage);
      }
        } catch (error: any) {
      console.error('❌ Registration error:', error);
      
      let errorMessage = 'เกิดข้อผิดพลาดในการเชื่อมต่อ กรุณาลองใหม่อีกครั้ง';
      
      if (error.name === 'AbortError') {
        errorMessage = 'การลงทะเบียนใช้เวลานานเกินไป กรุณาลองใหม่';
      } else if (error.message) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
      registrationRef.current = false;
      console.log('🔓 Registration process completed');
    }
  }, [personalInfo, router]);

  const handleFaceRegistrationError = useCallback((error: string) => {
    setError(error);
  }, []);

  // เพิ่ม callback สำหรับกลับไปแก้ไขข้อมูล
  const handleBack = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setError(null);
      // Reset registration state
      registrationRef.current = false;
    }
  }, [currentStep]);

  // เพิ่ม callback สำหรับจัดการ loading state
  const handleLoadingChange = useCallback((isLoading: boolean) => {
    setLoading(isLoading);
  }, []);

  // Reset registrationTriggered when กลับไป step 0
  useEffect(() => {
    if (currentStep === 0) {
      setRegistrationTriggered(false);
    }
  }, [currentStep]);

  // Handle form field changes for real-time validation
  const handleFieldChange = useCallback((field: string, value: string) => {
    setFormValues(prev => ({
      ...prev,
      [field]: value
    }));

    // Trigger appropriate availability checks
    switch (field) {
      case 'email':        emailCheck.check(value);
        break;
      case 'username':
        usernameCheck.check(value);
        break;
      case 'phone':
        if (value) {
          phoneCheck.check(value);
        }
        break;
      case 'firstName':
      case 'lastName':
        // ไม่ต้องตรวจสอบชื่อ
        break;
    }
  }, [formValues, emailCheck, usernameCheck, phoneCheck]);

  // Get status icon for field validation
  const getFieldStatus = (checkResult: { error?: string | null; available?: boolean; message?: string }, loading: boolean) => {
    if (loading) {
      return <LoadingOutlined style={{ color: '#1890ff' }} />;
    }
    if (checkResult.error) {
      return <CloseOutlined style={{ color: '#ff4d4f' }} />;
    }
    if (checkResult.available === false) {
      return <CloseOutlined style={{ color: '#ff4d4f' }} />;
    }
    if (checkResult.available === true && checkResult.message) {
      return <CheckOutlined style={{ color: '#52c41a' }} />;
    }
    return null;
  };

  const steps = [
    {
      title: 'ข้อมูลส่วนตัว',
      icon: <UserOutlined />,
      description: 'กรอกข้อมูลพื้นฐาน'
    },
    {
      title: 'สแกนใบหน้า',
      icon: <CameraOutlined />,
      description: 'ลงทะเบียนใบหน้าเพื่อความปลอดภัย'
    },
    {
      title: 'เสร็จสิ้น',
      icon: <CheckCircleOutlined />,
      description: 'สมัครสมาชิกสำเร็จ'
    }
  ];

  const renderPersonalInfoForm = () => (
    <Card className="max-w-2xl mx-auto">
      <div className="mb-8 text-center">
        <Title level={3}>ข้อมูลส่วนตัว</Title>
        <Paragraph type="secondary">
          กรอกข้อมูลพื้นฐานของคุณเพื่อสร้างบัญชี
        </Paragraph>
      </div>      {/* Status Alert */}
      {(emailCheck.loading || usernameCheck.loading || phoneCheck.loading) && (
        <Alert
          message="กำลังตรวจสอบความพร้อมใช้งานของข้อมูล..."
          type="info"
          showIcon
          icon={<LoadingOutlined />}
          className="mb-4"
        />
      )}

      {(!isFormValid() && (emailCheck.message || usernameCheck.message || phoneCheck.message || nameCheck.message)) && (
        <Alert
          message="พบข้อมูลที่ไม่พร้อมใช้งาน"
          description="กรุณาแก้ไขข้อมูลที่ทำเครื่องหมายสีแดงก่อนดำเนินการต่อ"
          type="warning"
          showIcon
          className="mb-4"
        />
      )}

      {isFormValid() && !emailCheck.loading && !usernameCheck.loading && !phoneCheck.loading && !nameCheck.loading && 
       (emailCheck.message || usernameCheck.message || nameCheck.message) && (
        <Alert
          message="ข้อมูลทั้งหมดพร้อมใช้งาน"
          description="คุณสามารถดำเนินการต่อได้"
          type="success"
          showIcon
          className="mb-4"
        />
      )}

      <Form
        form={form}
        layout="vertical"
        onFinish={handlePersonalInfoSubmit}
        autoComplete="off"
        size="large"
      >
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              name="firstName"
              label="ชื่อ"
              rules={[
                { required: true, message: 'กรุณากรอกชื่อ' },
                { min: 2, message: 'ชื่อต้องมีอย่างน้อย 2 ตัวอักษร' }
              ]}
              help={nameCheck.error || (!nameCheck.available && nameCheck.message) || (nameCheck.available && nameCheck.message)}
              validateStatus={
                nameCheck.loading ? 'validating' : 
                nameCheck.error ? 'error' : 
                !nameCheck.available ? 'error' : 
                nameCheck.available && nameCheck.message ? 'success' : 
                ''
              }
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="ชื่อ" 
                onChange={e => handleFieldChange('firstName', e.target.value)}
                addonAfter={getFieldStatus(nameCheck, nameCheck.loading)}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              name="lastName"
              label="นามสกุล"
              rules={[
                { required: true, message: 'กรุณากรอกนามสกุล' },
                { min: 2, message: 'นามสกุลต้องมีอย่างน้อย 2 ตัวอักษร' }
              ]}
              help={nameCheck.error || (!nameCheck.available && nameCheck.message) || (nameCheck.available && nameCheck.message)}
              validateStatus={
                nameCheck.loading ? 'validating' : 
                nameCheck.error ? 'error' : 
                !nameCheck.available ? 'error' : 
                nameCheck.available && nameCheck.message ? 'success' : 
                ''
              }
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="นามสกุล" 
                onChange={e => handleFieldChange('lastName', e.target.value)}
                addonAfter={getFieldStatus(nameCheck, nameCheck.loading)}
              />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          name="email"
          label="อีเมล"
          rules={[
            { required: true, message: 'กรุณากรอกอีเมล' },
            { type: 'email', message: 'รูปแบบอีเมลไม่ถูกต้อง' }
          ]}
          help={emailCheck.error || (!emailCheck.available && emailCheck.message) || (emailCheck.available && emailCheck.message)}
          validateStatus={
            emailCheck.loading ? 'validating' : 
            emailCheck.error ? 'error' : 
            !emailCheck.available ? 'error' : 
            emailCheck.available && emailCheck.message ? 'success' : 
            ''
          }
        >
          <Input 
            prefix={<MailOutlined />} 
            placeholder="example@email.com" 
            onChange={e => handleFieldChange('email', e.target.value)}
            addonAfter={getFieldStatus(emailCheck, emailCheck.loading)}
          />
        </Form.Item>

        <Form.Item
          name="username"
          label="ชื่อผู้ใช้"
          rules={[
            { required: true, message: 'กรุณากรอกชื่อผู้ใช้' },
            { min: 3, message: 'ชื่อผู้ใช้ต้องมีอย่างน้อย 3 ตัวอักษร' },
            { 
              pattern: /^[a-zA-Z0-9_]+$/, 
              message: 'ชื่อผู้ใช้ใช้ได้เฉพาะ a-z, A-Z, 0-9 และ _' 
            }
          ]}
          help={usernameCheck.error || (!usernameCheck.available && usernameCheck.message) || (usernameCheck.available && usernameCheck.message)}
          validateStatus={
            usernameCheck.loading ? 'validating' : 
            usernameCheck.error ? 'error' : 
            !usernameCheck.available ? 'error' : 
            usernameCheck.available && usernameCheck.message ? 'success' : 
            ''
          }
        >
          <Input 
            prefix={<UserOutlined />} 
            placeholder="username" 
            onChange={e => handleFieldChange('username', e.target.value)}
            addonAfter={getFieldStatus(usernameCheck, usernameCheck.loading)}
          />
        </Form.Item>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              name="password"
              label="รหัสผ่าน"
              rules={[
                { required: true, message: 'กรุณากรอกรหัสผ่าน' },
                { min: 6, message: 'รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร' }
              ]}
            >
              <Input.Password prefix={<LockOutlined />} placeholder="รหัสผ่าน" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              name="confirmPassword"
              label="ยืนยันรหัสผ่าน"
              dependencies={['password']}
              rules={[
                { required: true, message: 'กรุณายืนยันรหัสผ่าน' },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue('password') === value) {
                      return Promise.resolve();
                    }
                    return Promise.reject(new Error('รหัสผ่านไม่ตรงกัน'));
                  },
                }),
              ]}
            >
              <Input.Password prefix={<LockOutlined />} placeholder="ยืนยันรหัสผ่าน" />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          name="phone"
          label="เบอร์โทรศัพท์ (ไม่บังคับ)"
          help={phoneCheck.error || (!phoneCheck.available && phoneCheck.message) || (phoneCheck.available && phoneCheck.message)}
          validateStatus={
            phoneCheck.loading ? 'validating' : 
            phoneCheck.error ? 'error' : 
            !phoneCheck.available ? 'error' : 
            phoneCheck.available && phoneCheck.message ? 'success' : 
            ''
          }
        >
          <Input 
            prefix={<PhoneOutlined />} 
            placeholder="08X-XXX-XXXX" 
            onChange={e => handleFieldChange('phone', e.target.value)}
            addonAfter={getFieldStatus(phoneCheck, phoneCheck.loading)}
          />
        </Form.Item>

        <Form.Item
          name="dateOfBirth"
          label="วันเกิด (ไม่บังคับ)"
        >
          <DatePicker 
            style={{ width: '100%' }} 
            placeholder="เลือกวันเกิด"
            format="DD/MM/YYYY"
          />
        </Form.Item>

        <Form.Item
          name="agreeToTerms"
          valuePropName="checked"
          rules={[
            { 
              validator: (_, value) =>
                value ? Promise.resolve() : Promise.reject(new Error('กรุณายอมรับข้อตกลงการใช้งาน'))
            }
          ]}
        >
          <Checkbox>
            ฉันยอมรับ <a href="/terms" target="_blank">ข้อตกลงการใช้งาน</a> และ{' '}
            <a href="/privacy" target="_blank">นโยบายความเป็นส่วนตัว</a>
          </Checkbox>
        </Form.Item>

        <Form.Item>
          <Button 
            type="primary" 
            htmlType="submit" 
            loading={loading}
            size="large"
            block
            icon={<ArrowRightOutlined />}
          >
            ดำเนินการต่อ
          </Button>
        </Form.Item>

        <div className="text-center mt-4">
          <Text type="secondary">
            มีบัญชีอยู่แล้ว? <a href="/login">เข้าสู่ระบบ</a>
          </Text>
        </div>
      </Form>      {/* Debug info - remove in production */}
      {process.env.NODE_ENV === 'development' && (
        <div className="bg-gray-50 p-2 text-xs mb-4 rounded">
          <div>Email: {formValues.email} → {emailCheck.available ? '✓' : emailCheck.loading ? '⏳' : '✗'}</div>
          <div>Username: {formValues.username} → {usernameCheck.available ? '✓' : usernameCheck.loading ? '⏳' : '✗'}</div>
          <div>Phone: {formValues.phone || 'empty'} → {phoneCheck.available ? '✓' : phoneCheck.loading ? '⏳' : '✗'}</div>
          <div>Form Valid: {isFormValid() ? 'YES' : 'NO'}</div>
        </div>
      )}
    </Card>
  );

  // ปรับปรุง renderFaceRegistration ให้ส่ง props ที่ถูกต้อง
  const renderFaceRegistration = () => (
    <Card className="max-w-2xl mx-auto">
      <div className="mb-8 text-center">
        <Title level={3}>สแกนใบหน้าเพื่อความปลอดภัย</Title>
        <Paragraph type="secondary">
          กรุณาให้ใบหน้าอยู่ในกรอบและมองตรงเข้ากล้อง
        </Paragraph>
      </div>

      {/* Face API Status */}
      <FaceAPIStatus showDetails={true} />

      <FaceRegistration
        onComplete={handleFaceRegistrationComplete}
        onError={handleFaceRegistrationError}
        onBack={handleBack}
        userInfo={personalInfo ? {
          firstName: personalInfo.firstName,
          lastName: personalInfo.lastName,
          userId: '' // ยังไม่มี userId ในขณะนี้
        } : undefined}
        loading={loading}
        onLoadingChange={handleLoadingChange}
      />
    </Card>
  );

  const renderSuccess = () => (
    <Card className="max-w-2xl mx-auto text-center">
      <div className="mb-8">
        <CheckCircleOutlined className="text-6xl text-green-500 mb-4" />
        <Title level={2} className="text-green-600">สมัครสมาชิกสำเร็จ!</Title>
        <Paragraph className="text-lg">
          ยินดีต้อนรับสู่ FaceSocial คุณ{' '}
          <Text strong>{personalInfo?.firstName} {personalInfo?.lastName}</Text>
        </Paragraph>
      </div>

      <div className="bg-gray-50 p-6 rounded-lg mb-6">
        <Title level={4}>ขั้นตอนถัดไป:</Title>
        <div className="text-left">
          <p>✅ บัญชีของคุณถูกสร้างสำเร็จแล้ว</p>
          <p>📧 กรุณาตรวจสอบอีเมลเพื่อยืนยันบัญชี</p>
          <p>🔐 ระบบจดจำใบหน้าพร้อมใช้งานแล้ว</p>
        </div>
      </div>

      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Button type="primary" size="large" onClick={() => router.push('/login')}>
          เข้าสู่ระบบ
        </Button>
        <Button type="default" onClick={() => router.push('/')}>
          กลับสู่หน้าแรก
        </Button>
      </Space>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-blue-600">🤖 FaceSocial</h1>
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <Link href="/" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">
                  หน้าแรก
                </Link>
                <Link href="/login" className="text-blue-600 hover:text-blue-800 px-3 py-2 rounded-md text-sm font-medium">
                  เข้าสู่ระบบ
                </Link>
                <FaceAPIStatus />
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="py-12 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Steps Indicator */}
          <div className="mb-8">
            <Steps
              current={currentStep}
              items={steps}
              size="default"
              className="max-w-2xl mx-auto"
            />
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
              className="mb-6 max-w-2xl mx-auto"
            />
          )}

          {/* Loading Overlay - แสดงเฉพาะเมื่อมีการประมวลผล */}
          {loading && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg p-8 text-center max-w-md mx-4">
                <Spin size="large" />
                <div className="mt-4">
                  <Title level={4}>กำลังดำเนินการ...</Title>
                  <Text type="secondary">กรุณารอสักครู่</Text>
                </div>
              </div>
            </div>
          )}

          {/* Step Content */}
          {currentStep === 0 && renderPersonalInfoForm()}
          {currentStep === 1 && renderFaceRegistration()}
          {currentStep === 2 && renderSuccess()}
        </div>
      </div>

      {/* Debug information ใน development mode */}
      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-4 right-4 bg-gray-800 text-white p-3 rounded-lg text-xs max-w-xs">
          <div>Current Step: {currentStep}</div>
          <div>Loading: {loading ? 'Yes' : 'No'}</div>
          <div>Has Personal Info: {personalInfo ? 'Yes' : 'No'}</div>
          <div>Registration Ref: {registrationRef.current ? 'Active' : 'Inactive'}</div>
          {personalInfo && (
            <div>User: {personalInfo.firstName} {personalInfo.lastName}</div>
          )}
        </div>
      )}
    </div>
  );
};

export default RegisterPage;