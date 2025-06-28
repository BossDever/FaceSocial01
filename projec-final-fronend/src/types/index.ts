// Global Types for FaceSocial

export interface User {
  id: string;
  email: string;
  username: string;
  displayName: string;
  avatar?: string;
  bio?: string;
  isVerified: boolean;
  faceDataConsent: boolean;
  privacySettings: PrivacySettings;
  createdAt: Date;
  updatedAt: Date;
}

export interface PrivacySettings {
  profileVisibility: 'public' | 'friends' | 'private';
  faceRecognitionEnabled: boolean;
  aiAnalysisEnabled: boolean;
  dataRetentionPeriod: number; // days
  shareWithPartners: boolean;
}

export interface Post {
  id: string;
  authorId: string;
  author: User;
  content: string;
  images?: string[];
  aiAnalysis?: AIAnalysis;
  likes: number;
  comments: Comment[];
  shares: number;
  visibility: 'public' | 'friends' | 'private';
  createdAt: Date;
  updatedAt: Date;
}

export interface Comment {
  id: string;
  postId: string;
  authorId: string;
  author: User;
  content: string;
  likes: number;
  replies: Comment[];
  createdAt: Date;
}

export interface AIAnalysis {
  faceDetection?: FaceDetectionResult;
  objectDetection?: ObjectDetectionResult;
  contentModeration?: ContentModerationResult;
  deepfakeDetection?: DeepfakeDetectionResult;
}

export interface FaceDetectionResult {
  faces: DetectedFace[];
  processingTime: number;
  confidence: number;
}

export interface DetectedFace {
  id: string;
  boundingBox: BoundingBox;
  confidence: number;
  landmarks?: FaceLandmark[];
  attributes?: FaceAttributes;
  recognizedPerson?: RecognizedPerson;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceLandmark {
  type: string;
  x: number;
  y: number;
}

export interface FaceAttributes {
  age?: number;
  gender?: 'male' | 'female' | 'unknown';
  emotion?: Emotion;
  glasses?: boolean;
  beard?: boolean;
  mustache?: boolean;
}

export interface Emotion {
  happy: number;
  sad: number;
  angry: number;
  surprised: number;
  neutral: number;
  fear: number;
  disgust: number;
}

export interface RecognizedPerson {
  userId: string;
  confidence: number;
  relationship: 'friend' | 'family' | 'colleague' | 'unknown';
}

export interface ObjectDetectionResult {
  objects: DetectedObject[];
  processingTime: number;
}

export interface DetectedObject {
  class: string;
  confidence: number;
  boundingBox: BoundingBox;
}

export interface ContentModerationResult {
  safe: boolean;
  categories: {
    adult: number;
    violence: number;
    hate: number;
    selfHarm: number;
  };
  blockedReason?: string;
}

export interface DeepfakeDetectionResult {
  isDeepfake: boolean;
  confidence: number;
  analysisDetails: {
    faceConsistency: number;
    temporalConsistency: number;
    artifactDetection: number;
  };
}

export interface ChatMessage {
  id: string;
  chatId: string;
  senderId: string;
  sender: User;
  content: string;
  type: 'text' | 'image' | 'video' | 'file';
  fileUrl?: string;
  isRead: boolean;
  aiAnalyzed?: boolean;
  createdAt: Date;
}

export interface Chat {
  id: string;
  participants: User[];
  lastMessage?: ChatMessage;
  isGroup: boolean;
  groupName?: string;
  groupAvatar?: string;
  unreadCount: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface Notification {
  id: string;  userId: string;
  type: 'like' | 'comment' | 'share' | 'friend_request' | 'ai_analysis' | 'system';
  title: string;
  message: string;
  isRead: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data?: any;
  createdAt: Date;
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}
