'use client';

import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Avatar, 
  Typography, 
  Button, 
  Space,
  Input,
  Upload,
  Form,
  message,
  Divider,
  Empty,
  Spin,
  Modal,
  Row,
  Col,
  Tag,
  Tooltip
} from 'antd';
import { 
  UserOutlined, 
  CameraOutlined,
  SendOutlined,
  HeartOutlined,
  HeartFilled,
  MessageOutlined,
  ShareAltOutlined,
  PlusOutlined,
  PictureOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import type { UploadProps } from 'antd';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../components/layout/AppLayout';
import { cardStyle, colors, buttonStyle } from '../../styles/theme';

dayjs.extend(relativeTime);

const { Text, Paragraph } = Typography;
const { TextArea } = Input;

interface Post {
  id: string;
  userId: string;
  content: string;
  imageUrl?: string;
  createdAt: string;
  user: {
    id: string;
    username: string;
    firstName: string;
    lastName: string;
    profilePicture?: string;
  };
  likes: Array<{
    id: string;
    userId: string;
    user: {
      username: string;
    };
  }>;
  comments: Array<{
    id: string;
    userId: string;
    content: string;
    createdAt: string;
    user: {
      username: string;
      firstName: string;
      lastName: string;
      profilePicture?: string;
    };
  }>;
  _count: {
    likes: number;
    comments: number;
  };
}

interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  profilePicture?: string;
}

const FeedPage: React.FC = () => {
  const router = useRouter();
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [posting, setPosting] = useState(false);
  const [newPost, setNewPost] = useState('');
  const [user, setUser] = useState<User | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [commentModalVisible, setCommentModalVisible] = useState(false);
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [newComment, setNewComment] = useState('');

  useEffect(() => {
    fetchCurrentUser();
    fetchPosts();
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

  const fetchPosts = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const response = await fetch('/api/posts', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const postsData = await response.json();
        setPosts(postsData);
      } else {
        message.error('ไม่สามารถโหลดโพสต์ได้');
      }
    } catch (error) {
      console.error('Error fetching posts:', error);
      message.error('เกิดข้อผิดพลาดในการโหลดโพสต์');
    } finally {
      setLoading(false);
    }
  };

  const handleCreatePost = async () => {
    if (!newPost.trim() && !imageFile) {
      message.warning('กรุณาเพิ่มข้อความหรือรูปภาพ');
      return;
    }

    try {
      setPosting(true);
      const token = localStorage.getItem('token');
      
      const formData = new FormData();
      formData.append('content', newPost);
      if (imageFile) {
        formData.append('image', imageFile);
      }

      const response = await fetch('/api/posts', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        message.success('โพสต์สำเร็จแล้ว!');
        setNewPost('');
        setImageFile(null);
        fetchPosts(); // Refresh posts
      } else {
        const error = await response.json();
        message.error(error.message || 'ไม่สามารถโพสต์ได้');
      }
    } catch (error) {
      console.error('Error creating post:', error);
      message.error('เกิดข้อผิดพลาดในการโพสต์');
    } finally {
      setPosting(false);
    }
  };

  const handleLike = async (postId: string) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/posts/${postId}/like`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        fetchPosts(); // Refresh to get updated like count
      }
    } catch (error) {
      console.error('Error liking post:', error);
      message.error('เกิดข้อผิดพลาดในการกดไลค์');
    }
  };

  const handleComment = async (postId: string) => {
    if (!newComment.trim()) {
      message.warning('กรุณาเพิ่มข้อความในคอมเมนต์');
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/posts/${postId}/comments`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: newComment }),
      });

      if (response.ok) {
        message.success('เพิ่มคอมเมนต์สำเร็จ!');
        setNewComment('');
        setCommentModalVisible(false);
        fetchPosts(); // Refresh posts
      } else {
        message.error('ไม่สามารถเพิ่มคอมเมนต์ได้');
      }
    } catch (error) {
      console.error('Error adding comment:', error);
      message.error('เกิดข้อผิดพลาดในการเพิ่มคอมเมนต์');
    }
  };

  const uploadProps: UploadProps = {
    name: 'file',
    beforeUpload: (file) => {
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        message.error('กรุณาเลือกไฟล์รูปภาพเท่านั้น!');
        return false;
      }
      const isLt5M = file.size / 1024 / 1024 < 5;
      if (!isLt5M) {
        message.error('ขนาดไฟล์ต้องน้อยกว่า 5MB!');
        return false;
      }
      setImageFile(file);
      return false; // Prevent auto upload
    },
    onRemove: () => {
      setImageFile(null);
    },
    showUploadList: true,
    maxCount: 1,
  };

  // Mock sample posts if no posts available
  const samplePosts: Post[] = [
    {
      id: 'sample-1',
      userId: 'sample-user',
      content: 'ยินดีต้อนรับสู่ FaceSocial! 🎉 แชร์เรื่องราวและเชื่อมต่อกับเพื่อนๆ ได้ที่นี่',
      imageUrl: '/uploads/sample-sunset.jpg',
      createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      user: {
        id: 'sample-user',
        username: 'facesocial',
        firstName: 'Face',
        lastName: 'Social',
        profilePicture: '/images/avatars/admin.jpg'
      },
      likes: [],
      comments: [],
      _count: { likes: 12, comments: 3 }
    },
    {
      id: 'sample-2',
      userId: 'sample-user-2',
      content: 'สวัสดีทุกคน! มาร่วมสนุกกับฟีเจอร์ใหม่ๆ ของเรากันเถอะ 🚀',
      createdAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
      user: {
        id: 'sample-user-2',
        username: 'jane_doe',
        firstName: 'Jane',
        lastName: 'Doe',
        profilePicture: '/images/avatars/jane.jpg'
      },
      likes: [],
      comments: [],
      _count: { likes: 8, comments: 2 }
    }
  ];

  const displayPosts = posts.length > 0 ? posts : samplePosts;

  return (
    <AppLayout>
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Create Post Card */}
        <Card style={cardStyle} className="shadow-sm">
          <div className="flex items-start space-x-3">
            <Avatar 
              src={user?.profilePicture} 
              icon={<UserOutlined />}
              size={48}
            />
            <div className="flex-1">
              <TextArea
                placeholder="คุณกำลังคิดอะไรอยู่?"
                value={newPost}
                onChange={(e) => setNewPost(e.target.value)}
                rows={3}
                className="border-0 bg-gray-50 rounded-lg resize-none focus:bg-white"
                style={{ 
                  fontSize: '16px',
                  borderRadius: '12px'
                }}
              />
              
              {imageFile && (
                <div className="mt-3 p-2 bg-gray-50 rounded-lg">
                  <Text className="text-gray-600 text-sm">📷 {imageFile.name}</Text>
                </div>
              )}

              <div className="flex justify-between items-center mt-4">
                <Upload {...uploadProps}>
                  <Button 
                    icon={<PictureOutlined />} 
                    type="text"
                    className="text-gray-600 hover:text-blue-600"
                  >
                    เพิ่มรูปภาพ
                  </Button>
                </Upload>
                
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleCreatePost}
                  loading={posting}
                  disabled={!newPost.trim() && !imageFile}
                  style={buttonStyle.primary}
                  className="px-6"
                >
                  โพสต์
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* Posts Feed */}
        {loading ? (
          <div className="text-center py-12">
            <Spin size="large" />
            <Text className="block mt-4 text-gray-600">กำลังโหลดโพสต์...</Text>
          </div>
        ) : displayPosts.length > 0 ? (
          <div className="space-y-6">
            {displayPosts.map((post) => (
              <Card key={post.id} style={cardStyle} className="shadow-sm hover:shadow-md transition-shadow">
                {/* Post Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <Avatar 
                      src={post.user.profilePicture} 
                      icon={<UserOutlined />}
                      size={48}
                    />
                    <div>
                      <Text strong className="text-gray-800 block">
                        {post.user.firstName} {post.user.lastName}
                      </Text>
                      <div className="flex items-center space-x-2 text-gray-500 text-sm">
                        <Text className="text-gray-500">@{post.user.username}</Text>
                        <span>•</span>
                        <div className="flex items-center space-x-1">
                          <ClockCircleOutlined className="text-xs" />
                          <span>{dayjs(post.createdAt).fromNow()}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Post Content */}
                <div className="mb-4">
                  <Paragraph className="text-gray-800 text-base mb-3 whitespace-pre-wrap">
                    {post.content}
                  </Paragraph>
                  
                  {post.imageUrl && (
                    <div className="rounded-lg overflow-hidden bg-gray-100">
                      <img 
                        src={post.imageUrl} 
                        alt="Post"
                        className="w-full h-auto max-h-96 object-cover"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    </div>
                  )}
                </div>

                {/* Post Actions */}
                <div className="flex items-center justify-between pt-3 border-t border-gray-100">
                  <div className="flex items-center space-x-6">
                    <Tooltip title="ไลค์">
                      <Button
                        type="text"
                        icon={<HeartOutlined className="text-lg" />}
                        onClick={() => handleLike(post.id)}
                        className="flex items-center space-x-2 text-gray-600 hover:text-red-500 transition-colors"
                      >
                        <span>{post._count.likes}</span>
                      </Button>
                    </Tooltip>

                    <Tooltip title="แสดงความคิดเห็น">
                      <Button
                        type="text"
                        icon={<MessageOutlined className="text-lg" />}
                        onClick={() => {
                          setSelectedPost(post);
                          setCommentModalVisible(true);
                        }}
                        className="flex items-center space-x-2 text-gray-600 hover:text-blue-500 transition-colors"
                      >
                        <span>{post._count.comments}</span>
                      </Button>
                    </Tooltip>

                    <Tooltip title="แชร์">
                      <Button
                        type="text"
                        icon={<ShareAltOutlined className="text-lg" />}
                        onClick={() => message.info('ฟีเจอร์แชร์กำลังพัฒนา')}
                        className="text-gray-600 hover:text-green-500 transition-colors"
                      />
                    </Tooltip>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <Card style={cardStyle} className="text-center py-12">
            <Empty
              description={
                <div>
                  <Text className="text-gray-500 text-lg block mb-2">
                    ยังไม่มีโพสต์
                  </Text>
                  <Text className="text-gray-400">
                    เป็นคนแรกที่แชร์เรื่องราวในฟีดนี้!
                  </Text>
                </div>
              }
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          </Card>
        )}

        {/* Comment Modal */}
        <Modal
          title={
            <div className="flex items-center space-x-2">
              <MessageOutlined />
              <span>ความคิดเห็น</span>
            </div>
          }
          open={commentModalVisible}
          onCancel={() => {
            setCommentModalVisible(false);
            setNewComment('');
          }}
          footer={null}
          width={600}
        >
          {selectedPost && (
            <div className="space-y-4">
              {/* Original Post */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center space-x-3 mb-3">
                  <Avatar 
                    src={selectedPost.user.profilePicture} 
                    icon={<UserOutlined />}
                    size={32}
                  />
                  <div>
                    <Text strong>
                      {selectedPost.user.firstName} {selectedPost.user.lastName}
                    </Text>
                    <Text className="text-gray-500 text-sm block">
                      @{selectedPost.user.username}
                    </Text>
                  </div>
                </div>
                <Text>{selectedPost.content}</Text>
              </div>

              {/* Add Comment */}
              <div className="flex space-x-3">
                <Avatar 
                  src={user?.profilePicture} 
                  icon={<UserOutlined />}
                  size={40}
                />
                <div className="flex-1">
                  <TextArea
                    placeholder="เพิ่มความคิดเห็น..."
                    value={newComment}
                    onChange={(e) => setNewComment(e.target.value)}
                    rows={3}
                    className="border-gray-200"
                  />
                  <div className="flex justify-end mt-3">
                    <Button
                      type="primary"
                      onClick={() => handleComment(selectedPost.id)}
                      disabled={!newComment.trim()}
                      style={buttonStyle.primary}
                    >
                      ส่งความคิดเห็น
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </Modal>
      </div>
    </AppLayout>
  );
};

export default FeedPage;
