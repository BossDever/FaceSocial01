'use client';

import React, { useState } from 'react';
import {
  Modal,
  Form,
  Input,
  Button,
  Upload,
  Space,
  Switch,
  Tag,
  Avatar,
  message
} from 'antd';
import {
  PlusOutlined,
  EnvironmentOutlined,
  UserOutlined,
  TagOutlined
} from '@ant-design/icons';
import type { UploadProps, UploadFile } from 'antd';
import FaceTagModal from './FaceTagModal';

const { TextArea } = Input;

interface PostCreateModalProps {
  visible: boolean;
  onCancel: () => void;
  onSuccess: (post: any) => void;
}

interface TaggedUser {
  id: number;
  username: string;
  fullName: string;
  avatarUrl?: string;
  confidence: number;
  faceRegion: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

const PostCreateModal: React.FC<PostCreateModalProps> = ({
  visible,
  onCancel,
  onSuccess
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [selectedTags, setSelectedTags] = useState<TaggedUser[]>([]);
  const [showFaceTagModal, setShowFaceTagModal] = useState(false);
  const [uploadedImageFile, setUploadedImageFile] = useState<File | null>(null);

  const handleSubmit = async (values: any) => {
    try {
      setLoading(true);
      
      const token = localStorage.getItem('token');
      if (!token) {
        message.error('กรุณาเข้าสู่ระบบก่อน');
        return;
      }

      // Step 1: Create the post first
      const formData = new FormData();
      formData.append('content', values.content || '');
      formData.append('location', values.location || '');
      formData.append('isPublic', values.isPublic !== false ? 'true' : 'false');

      if (fileList[0]?.originFileObj) {
        formData.append('image', fileList[0].originFileObj);
      }

      const response = await fetch('/api/posts/create', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        
        // Step 2: If there's an image and we have tagged users, save the tags
        if (result.data && selectedTags.length > 0) {
          await saveFaceTags(result.data.id, selectedTags);
        }

        message.success('โพสต์สำเร็จ!');
        onSuccess(result.data);
        handleReset();
      } else {
        const error = await response.json();
        message.error(error.message || 'เกิดข้อผิดพลาดในการโพสต์');
      }
    } catch (error) {
      console.error('Error creating post:', error);
      message.error('เกิดข้อผิดพลาดในการโพสต์');
    } finally {
      setLoading(false);
    }
  };

  const saveFaceTags = async (postId: number, tags: TaggedUser[]) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/posts/face-tags', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          postId,
          tags: tags.map(tag => ({
            userId: tag.id,
            x: tag.faceRegion.x,
            y: tag.faceRegion.y,
            width: tag.faceRegion.width,
            height: tag.faceRegion.height
          }))
        })
      });

      if (response.ok) {
        console.log('Face tags saved successfully');
        message.success(`แท็ก ${tags.length} คนสำเร็จ!`);
      } else {
        console.error('Failed to save face tags');
        message.warning('บันทึกแท็กไม่สำเร็จ');
      }
    } catch (error) {
      console.error('Error saving face tags:', error);
      message.warning('เกิดข้อผิดพลาดในการบันทึกแท็ก');
    }
  };

  const handleUploadChange: UploadProps['onChange'] = ({ fileList: newFileList }) => {
    setFileList(newFileList);
    
    // If a new image is uploaded, store it for face tagging
    if (newFileList.length > 0 && newFileList[0].originFileObj) {
      setUploadedImageFile(newFileList[0].originFileObj);
    } else {
      setUploadedImageFile(null);
      setSelectedTags([]);
    }
  };

  const handleTagFaces = () => {
    if (uploadedImageFile) {
      setShowFaceTagModal(true);
    } else {
      message.warning('กรุณาเลือกรูปภาพก่อน');
    }
  };

  const handleTagsSelected = (tags: TaggedUser[]) => {
    setSelectedTags(tags);
    setShowFaceTagModal(false);
    if (tags.length > 0) {
      message.success(`เลือกแท็ก ${tags.length} คน`);
    }
  };

  const handleRemoveTag = (tagId: number) => {
    setSelectedTags(prev => prev.filter(tag => tag.id !== tagId));
  };

  const handleReset = () => {
    form.resetFields();
    setFileList([]);
    setSelectedTags([]);
    setUploadedImageFile(null);
    setShowFaceTagModal(false);
  };

  const handleCancel = () => {
    handleReset();
    onCancel();
  };

  const uploadButton = (
    <div>
      <PlusOutlined />
      <div style={{ marginTop: 8 }}>เลือกรูปภาพ</div>
    </div>
  );

  return (
    <>
      <Modal
        title="สร้างโพสต์ใหม่"
        open={visible}
        onCancel={handleCancel}
        width={600}
        footer={[
          <Button key="cancel" onClick={handleCancel}>
            ยกเลิก
          </Button>,
          <Button
            key="submit"
            type="primary"
            loading={loading}
            onClick={() => form.submit()}
          >
            โพสต์
          </Button>
        ]}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
        >
          <Form.Item
            name="content"
            label="เนื้อหา"
          >
            <TextArea
              rows={4}
              placeholder="คุณกำลังคิดอะไรอยู่?"
              maxLength={500}
              showCount
            />
          </Form.Item>

          <Form.Item label="รูปภาพ">
            <Upload
              listType="picture-card"
              fileList={fileList}
              onChange={handleUploadChange}
              beforeUpload={() => false}
              maxCount={1}
              accept="image/*"
            >
              {fileList.length === 0 && uploadButton}
            </Upload>
          </Form.Item>

          {/* Face Tagging Section */}
          {uploadedImageFile && (
            <Form.Item label="แท็กเพื่อน">
              <Space direction="vertical" style={{ width: '100%' }}>                <Button 
                  icon={<TagOutlined />} 
                  onClick={handleTagFaces}
                  type="dashed"
                  block
                >
                  🤖 แท็กเพื่อนอัตโนมัติ (AI ตรวจจับใบหน้า)
                </Button>
                
                {selectedTags.length > 0 && (
                  <div>
                    <div className="mb-2 text-sm text-gray-600">
                      แท็กแล้ว {selectedTags.length} คน:
                    </div>
                    <div className="space-x-1">
                      {selectedTags.map(tag => (
                        <Tag
                          key={tag.id}
                          closable
                          onClose={() => handleRemoveTag(tag.id)}
                          color="blue"
                        >
                          <Space size="small">
                            <Avatar
                              src={tag.avatarUrl}
                              icon={<UserOutlined />}
                              size="small"
                            />
                            {tag.fullName}
                          </Space>
                        </Tag>
                      ))}
                    </div>
                  </div>
                )}
              </Space>
            </Form.Item>
          )}

          <Form.Item
            name="location"
            label="สถานที่"
          >
            <Input
              prefix={<EnvironmentOutlined />}
              placeholder="เพิ่มสถานที่"
              maxLength={100}
            />
          </Form.Item>

          <Form.Item
            name="isPublic"
            label="การมองเห็น"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch
              checkedChildren="สาธารณะ"
              unCheckedChildren="เพื่อนเท่านั้น"
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* Face Tag Modal */}
      {showFaceTagModal && uploadedImageFile && (
        <FaceTagModal
          visible={showFaceTagModal}
          onClose={() => setShowFaceTagModal(false)}
          onTagsSelected={handleTagsSelected}
          imageFile={uploadedImageFile}
        />
      )}
    </>
  );
};

export default PostCreateModal;
