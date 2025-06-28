// Types for FaceSocial Database Models
// TypeScript interfaces สำหรับฐานข้อมูล

export interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  phone?: string;
  dateOfBirth?: Date;
  profileImageUrl?: string;
  isActive: boolean;
  isVerified: boolean;
  emailVerifiedAt?: Date;
  phoneVerifiedAt?: Date;
  createdAt: Date;
  updatedAt: Date;
  lastLoginAt?: Date;
}

export interface UserProfile {
  id: string;
  userId: string;
  bio?: string;
  website?: string;
  location?: string;
  work?: string;
  education?: string;
  relationshipStatus?: string;
  interests?: string[];
  coverImageUrl?: string;
  visibilitySettings: Record<string, any>;
  socialLinks: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

export interface Post {
  id: string;
  userId: string;
  content?: string;
  imageUrls?: string[];
  videoUrl?: string;
  location?: string;
  privacyLevel: 'public' | 'friends' | 'private';
  isArchived: boolean;
  likesCount: number;
  commentsCount: number;
  sharesCount: number;
  createdAt: Date;
  updatedAt: Date;
  
  // Relations
  user?: User;
  likes?: PostLike[];
  comments?: PostComment[];
  faceTags?: PostFaceTag[];
}

export interface PostFaceTag {
  id: string;
  postId: string;
  taggedUserId: string;
  taggerUserId: string;
  imageUrl: string;
  faceBbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  faceEmbedding?: number[];
  confidenceScore: number;
  isConfirmed: boolean;
  status: 'pending' | 'approved' | 'rejected';
  createdAt: Date;
  updatedAt: Date;
  
  // Relations
  taggedUser?: User;
  taggerUser?: User;
  post?: Post;
}

export interface PostLike {
  id: string;
  postId: string;
  userId: string;
  createdAt: Date;
  
  // Relations
  user?: User;
  post?: Post;
}

export interface PostComment {
  id: string;
  postId: string;
  userId: string;
  parentCommentId?: string;
  content: string;
  imageUrl?: string;
  likesCount: number;
  repliesCount: number;
  isEdited: boolean;
  createdAt: Date;
  updatedAt: Date;
  
  // Relations
  user?: User;
  post?: Post;
  parentComment?: PostComment;
  replies?: PostComment[];
  likes?: CommentLike[];
}

export interface CommentLike {
  id: string;
  commentId: string;
  userId: string;
  createdAt: Date;
  
  // Relations
  user?: User;
  comment?: PostComment;
}

export interface UserConnection {
  id: string;
  requesterId: string;
  addresseeId: string;
  status: 'pending' | 'accepted' | 'rejected' | 'blocked';
  createdAt: Date;
  updatedAt: Date;
  
  // Relations
  requester?: User;
  addressee?: User;
}

export interface Conversation {
  id: string;
  type: 'direct' | 'group';
  name?: string;
  description?: string;
  imageUrl?: string;
  createdBy?: string;
  isActive: boolean;
  lastMessageAt?: Date;
  createdAt: Date;
  updatedAt: Date;
  
  // Relations
  creator?: User;
  participants?: ConversationParticipant[];
  messages?: Message[];
}

export interface ConversationParticipant {
  id: string;
  conversationId: string;
  userId: string;
  role: 'admin' | 'member';
  joinedAt: Date;
  leftAt?: Date;
  isMuted: boolean;
  lastReadAt?: Date;
  
  // Relations
  user?: User;
  conversation?: Conversation;
}

export interface Message {
  id: string;
  conversationId: string;
  senderId: string;
  content?: string;
  messageType: 'text' | 'image' | 'video' | 'file' | 'face_tag';
  fileUrl?: string;
  fileType?: string;
  fileSize?: number;
  replyToMessageId?: string;
  isEdited: boolean;
  isDeleted: boolean;
  createdAt: Date;
  updatedAt: Date;
  
  // Relations
  sender?: User;
  conversation?: Conversation;
  replyToMessage?: Message;
  faceTags?: MessageFaceTag[];
  reactions?: MessageReaction[];
}

export interface MessageFaceTag {
  id: string;
  messageId: string;
  taggedUserId: string;
  faceBbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidenceScore: number;
  createdAt: Date;
  
  // Relations
  message?: Message;
  taggedUser?: User;
}

export interface MessageReaction {
  id: string;
  messageId: string;
  userId: string;
  reaction: string;
  createdAt: Date;
  
  // Relations
  message?: Message;
  user?: User;
}

export interface UserActivity {
  id: string;
  userId: string;
  activityType: string;
  targetType?: string;
  targetId?: string;
  metadata: Record<string, any>;
  createdAt: Date;
  
  // Relations
  user?: User;
}

export interface Notification {
  id: string;
  userId: string;
  fromUserId?: string;
  type: 'face_tag' | 'like' | 'comment' | 'friend_request' | 'message';
  title: string;
  content?: string;
  relatedType?: string;
  relatedId?: string;
  isRead: boolean;
  createdAt: Date;
  
  // Relations
  user?: User;
  fromUser?: User;
}

export interface FaceTrainingData {
  id: string;
  userId: string;
  imageUrl: string;
  faceBbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  embeddingVector: number[];
  qualityScore: number;
  isVerified: boolean;
  sourceType: 'upload' | 'post' | 'profile';
  createdAt: Date;
  
  // Relations
  user?: User;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrevious: boolean;
  };
}

// Request Types
export interface CreatePostRequest {
  content?: string;
  imageUrls?: string[];
  videoUrl?: string;
  location?: string;
  privacyLevel?: 'public' | 'friends' | 'private';
}

export interface CreateCommentRequest {
  postId: string;
  content: string;
  parentCommentId?: string;
  imageUrl?: string;
}

export interface SendMessageRequest {
  conversationId: string;
  content?: string;
  messageType?: 'text' | 'image' | 'video' | 'file';
  fileUrl?: string;
  replyToMessageId?: string;
}

export interface CreateFaceTagRequest {
  postId?: string;
  messageId?: string;
  taggedUserId: string;
  imageUrl: string;
  faceBbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidenceScore?: number;
}

export interface UpdateProfileRequest {
  bio?: string;
  website?: string;
  location?: string;
  work?: string;
  education?: string;
  relationshipStatus?: string;
  interests?: string[];
  coverImageUrl?: string;
  visibilitySettings?: Record<string, any>;
  socialLinks?: Record<string, any>;
}

// Feed Types
export interface FeedItem {
  id: string;
  type: 'post' | 'face_tag' | 'friend_activity';
  post?: Post;
  activity?: UserActivity;
  createdAt: Date;
}

export interface FeedRequest {
  page?: number;
  limit?: number;
  type?: 'all' | 'friends' | 'public';
}

// Chat Types
export interface ChatPreview {
  conversation: Conversation;
  lastMessage?: Message;
  unreadCount: number;
  otherParticipants: User[];
}

export interface CreateConversationRequest {
  type: 'direct' | 'group';
  participantIds: string[];
  name?: string;
  description?: string;
}

// Search Types
export interface SearchRequest {
  query: string;
  type?: 'users' | 'posts' | 'all';
  page?: number;
  limit?: number;
}

export interface SearchResult {
  users: User[];
  posts: Post[];
  total: number;
}

// Face Recognition Types
export interface FaceDetectionResult {
  faces: Array<{
    bbox: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
    confidence: number;
    embedding?: number[];
    recognizedUser?: User;
    similarity?: number;
  }>;
  imageUrl: string;
}

export interface FaceRecognitionRequest {
  imageUrl: string;
  threshold?: number;
  includeEmbedding?: boolean;
}
