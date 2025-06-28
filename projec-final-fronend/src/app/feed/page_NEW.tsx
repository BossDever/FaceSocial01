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
  Tooltip,
  Badge
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
  ClockCircleOutlined,
  EyeOutlined,
  MoreOutlined
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import type { UploadProps } from 'antd';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import AppLayout from '../../components/layout/AppLayout';
import { colors, cardStyle, buttonStyle, iconStyle, textStyle } from '../../styles/theme';

dayjs.extend(relativeTime);

const { Text, Paragraph, Title } = Typography;
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
  const [user, setUser] = useState<User | null>(null);
  const [newPostContent, setNewPostContent] = useState('');
  const [newPostImage, setNewPostImage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [commentModals, setCommentModals] = useState<{[key: string]: boolean}>({});
  const [commentInputs, setCommentInputs] = useState<{[key: string]: string}>({});
  const [commentLoading, setCommentLoading] = useState<{[key: string]: boolean}>({});

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
      }
    } catch (error) {
      console.error('Error fetching user:', error);
    }
  };

  const fetchPosts = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      const response = await fetch('/api/posts', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setPosts(data);
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
    if (!newPostContent.trim()) {
      message.warning('กรุณาใส่เนื้อหาโพสต์');
      return;
    }

    try {
      setSubmitting(true);
      const token = localStorage.getItem('token');

      const response = await fetch('/api/posts', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: newPostContent,
          imageUrl: newPostImage,
        }),
      });

      if (response.ok) {
        message.success('สร้างโพสต์สำเร็จ');
        setNewPostContent('');
        setNewPostImage(null);
        fetchPosts();
      } else {
        message.error('ไม่สามารถสร้างโพสต์ได้');
      }
    } catch (error) {
      console.error('Error creating post:', error);
      message.error('เกิดข้อผิดพลาดในการสร้างโพสต์');
    } finally {
      setSubmitting(false);
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
        fetchPosts(); // Refresh posts to update like count
      }
    } catch (error) {
      console.error('Error liking post:', error);
    }
  };

  const handleComment = async (postId: string) => {
    const content = commentInputs[postId];
    if (!content?.trim()) return;

    try {
      setCommentLoading({ ...commentLoading, [postId]: true });
      const token = localStorage.getItem('token');

      const response = await fetch(`/api/posts/${postId}/comments`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: content.trim() }),
      });

      if (response.ok) {
        setCommentInputs({ ...commentInputs, [postId]: '' });
        fetchPosts(); // Refresh posts to update comments
        message.success('แสดงความคิดเห็นสำเร็จ');
      }
    } catch (error) {
      console.error('Error commenting:', error);
      message.error('เกิดข้อผิดพลาดในการแสดงความคิดเห็น');
    } finally {
      setCommentLoading({ ...commentLoading, [postId]: false });
    }
  };

  const PostCard: React.FC<{ post: Post }> = ({ post }) => {
    const isLiked = post.likes.some(like => like.userId === user?.id);
    const showComments = commentModals[post.id] || false;

    return (
      <Card
        style={{
          ...cardStyle,
          marginBottom: '20px',
          border: `1px solid ${colors.light.borderPrimary}`,
          borderRadius: '16px',
          overflow: 'hidden',
          backgroundColor: colors.light.bgContainer,
        }}
        bodyStyle={{ padding: '20px' }}
      >
        {/* Post Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', marginBottom: '16px' }}>
          <Avatar
            size={48}
            src={post.user?.profilePicture}
            icon={<UserOutlined />}
            style={{ marginRight: '12px', cursor: 'pointer' }}
            onClick={() => router.push(`/profile/${post.user?.id}`)}
          />
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <Text 
                  strong 
                  style={{ 
                    ...textStyle.body,
                    fontWeight: 600,
                    cursor: 'pointer',
                    color: colors.light.textPrimary
                  }}
                  onClick={() => router.push(`/profile/${post.user?.id}`)}
                >
                  {post.user?.firstName} {post.user?.lastName}
                </Text>
                <br />
                <Text 
                  style={{ 
                    ...textStyle.caption,
                    color: colors.light.textSecondary 
                  }}
                >
                  @{post.user?.username} · {dayjs(post.createdAt).fromNow()}
                </Text>
              </div>
              <Button
                type="text"
                icon={<MoreOutlined style={iconStyle.secondary} />}
                style={{ color: colors.light.textTertiary }}
              />
            </div>
          </div>
        </div>

        {/* Post Content */}
        <div style={{ marginBottom: '16px' }}>
          <Paragraph 
            style={{ 
              ...textStyle.body,
              marginBottom: '12px',
              lineHeight: 1.6,
              color: colors.light.textPrimary
            }}
          >
            {post.content}
          </Paragraph>
          
          {post.imageUrl && (
            <div style={{ 
              borderRadius: '12px', 
              overflow: 'hidden',
              border: `1px solid ${colors.light.borderSecondary}`,
              cursor: 'pointer'
            }}>
              <img 
                src={post.imageUrl} 
                alt="Post image" 
                style={{ 
                  width: '100%', 
                  height: 'auto',
                  maxHeight: '400px',
                  objectFit: 'cover'
                }}
              />
            </div>
          )}
        </div>

        {/* Post Stats */}
        {(post._count.likes > 0 || post._count.comments > 0) && (
          <div style={{ 
            padding: '12px 0',
            borderTop: `1px solid ${colors.light.borderSecondary}`,
            borderBottom: `1px solid ${colors.light.borderSecondary}`,
            marginBottom: '16px'
          }}>
            <Space split={<Divider type="vertical" />}>
              {post._count.likes > 0 && (
                <Text style={{ ...textStyle.caption, color: colors.light.textSecondary }}>
                  <HeartFilled style={{ color: colors.light.error, marginRight: '4px' }} />
                  {post._count.likes} คนถูกใจ
                </Text>
              )}
              {post._count.comments > 0 && (
                <Text style={{ ...textStyle.caption, color: colors.light.textSecondary }}>
                  <MessageOutlined style={{ marginRight: '4px' }} />
                  {post._count.comments} ความคิดเห็น
                </Text>
              )}
            </Space>
          </div>
        )}

        {/* Post Actions */}
        <div style={{ display: 'flex', justifyContent: 'space-around' }}>
          <Button
            type="text"
            icon={isLiked ? 
              <HeartFilled style={{ ...iconStyle.error }} /> : 
              <HeartOutlined style={iconStyle.secondary} />
            }
            onClick={() => handleLike(post.id)}
            style={{
              color: isLiked ? colors.light.error : colors.light.textSecondary,
              fontWeight: isLiked ? 600 : 500,
            }}
          >
            ถูกใจ
          </Button>
          
          <Button
            type="text"
            icon={<MessageOutlined style={iconStyle.secondary} />}
            onClick={() => setCommentModals({ ...commentModals, [post.id]: !showComments })}
            style={{ color: colors.light.textSecondary }}
          >
            แสดงความคิดเห็น
          </Button>
          
          <Button
            type="text"
            icon={<ShareAltOutlined style={iconStyle.secondary} />}
            style={{ color: colors.light.textSecondary }}
            onClick={() => message.info('ฟีเจอร์แชร์กำลังพัฒนา')}
          >
            แชร์
          </Button>
        </div>

        {/* Comments Section */}
        {showComments && (
          <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: `1px solid ${colors.light.borderSecondary}` }}>
            {/* Comment Input */}
            <div style={{ display: 'flex', marginBottom: '16px' }}>
              <Avatar
                size={32}
                src={user?.profilePicture}
                icon={<UserOutlined />}
                style={{ marginRight: '8px' }}
              />
              <div style={{ flex: 1 }}>
                <Input.TextArea
                  value={commentInputs[post.id] || ''}
                  onChange={(e) => setCommentInputs({ ...commentInputs, [post.id]: e.target.value })}
                  placeholder="เขียนความคิดเห็น..."
                  autoSize={{ minRows: 1, maxRows: 3 }}
                  style={{
                    backgroundColor: colors.light.bgSecondary,
                    border: `1px solid ${colors.light.borderSecondary}`,
                    borderRadius: '20px',
                    fontSize: '14px',
                  }}
                />
                <div style={{ marginTop: '8px', textAlign: 'right' }}>
                  <Button
                    type="primary"
                    size="small"
                    icon={<SendOutlined />}
                    loading={commentLoading[post.id]}
                    onClick={() => handleComment(post.id)}
                    disabled={!commentInputs[post.id]?.trim()}
                    style={buttonStyle.primary}
                  >
                    ส่ง
                  </Button>
                </div>
              </div>
            </div>

            {/* Comments List */}
            {post.comments.length > 0 ? (
              <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {post.comments.map((comment) => (
                  <div key={comment.id} style={{ display: 'flex', marginBottom: '12px' }}>
                    <Avatar
                      size={32}
                      src={comment.user?.profilePicture}
                      icon={<UserOutlined />}
                      style={{ marginRight: '8px' }}
                    />
                    <div style={{ 
                      flex: 1,
                      backgroundColor: colors.light.bgSecondary,
                      padding: '8px 12px',
                      borderRadius: '16px',
                    }}>
                      <Text 
                        strong 
                        style={{ 
                          ...textStyle.caption,
                          fontWeight: 600,
                          color: colors.light.textPrimary 
                        }}
                      >
                        {comment.user?.firstName} {comment.user?.lastName}
                      </Text>
                      <Text 
                        style={{ 
                          ...textStyle.caption,
                          color: colors.light.textSecondary,
                          marginLeft: '8px'
                        }}
                      >
                        {dayjs(comment.createdAt).fromNow()}
                      </Text>
                      <div style={{ marginTop: '4px' }}>
                        <Text style={{ ...textStyle.body, color: colors.light.textPrimary }}>
                          {comment.content}
                        </Text>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <Text style={{ ...textStyle.caption, color: colors.light.textTertiary }}>
                ยังไม่มีความคิดเห็น
              </Text>
            )}
          </div>
        )}
      </Card>
    );
  };

  if (loading) {
    return (
      <AppLayout>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          minHeight: '400px' 
        }}>
          <Spin size="large" />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div style={{ maxWidth: '600px', margin: '0 auto' }}>
        {/* Create Post Card */}
        <Card
          style={{
            ...cardStyle,
            marginBottom: '24px',
            backgroundColor: colors.light.bgContainer,
            borderRadius: '16px',
          }}
          bodyStyle={{ padding: '20px' }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <Avatar
              size={48}
              src={user?.profilePicture}
              icon={<UserOutlined />}
              style={{ marginRight: '12px' }}
            />
            <div style={{ flex: 1 }}>
              <TextArea
                value={newPostContent}
                onChange={(e) => setNewPostContent(e.target.value)}
                placeholder="คุณกำลังคิดอะไรอยู่?"
                autoSize={{ minRows: 3, maxRows: 6 }}
                style={{
                  backgroundColor: colors.light.bgSecondary,
                  border: `1px solid ${colors.light.borderSecondary}`,
                  borderRadius: '12px',
                  fontSize: '16px',
                  lineHeight: 1.6,
                }}
              />
              
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                marginTop: '16px'
              }}>
                <Button
                  type="text"
                  icon={<PictureOutlined style={iconStyle.secondary} />}
                  style={{ color: colors.light.textSecondary }}
                >
                  รูปภาพ
                </Button>
                
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  loading={submitting}
                  onClick={handleCreatePost}
                  disabled={!newPostContent.trim()}
                  style={buttonStyle.primary}
                >
                  โพสต์
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* Posts List */}
        {posts.length > 0 ? (
          posts.map((post) => (
            <PostCard key={post.id} post={post} />
          ))
        ) : (
          <Card style={cardStyle}>
            <Empty 
              description={
                <Text style={{ color: colors.light.textSecondary }}>
                  ยังไม่มีโพสต์
                </Text>
              }
            />
          </Card>
        )}
      </div>
    </AppLayout>
  );
};

export default FeedPage;
