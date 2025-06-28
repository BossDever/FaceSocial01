import { create } from 'zustand';
import { AIAnalysis } from '@/types';

interface AIState {
  isProcessing: boolean;
  currentAnalysis: AIAnalysis | null;
  analysisHistory: AIAnalysis[];
  faceRecognitionEnabled: boolean;
  confidenceThreshold: number;
  
  // Actions
  setProcessing: (processing: boolean) => void;
  setAnalysis: (analysis: AIAnalysis) => void;
  addToHistory: (analysis: AIAnalysis) => void;
  clearHistory: () => void;
  toggleFaceRecognition: () => void;
  setConfidenceThreshold: (threshold: number) => void;
}

export const useAIStore = create<AIState>((set) => ({
  isProcessing: false,
  currentAnalysis: null,
  analysisHistory: [],
  faceRecognitionEnabled: true,
  confidenceThreshold: 0.7,
  
  setProcessing: (processing) => set({ isProcessing: processing }),
  setAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  addToHistory: (analysis) => 
    set((state) => ({ 
      analysisHistory: [analysis, ...state.analysisHistory].slice(0, 50) // Keep last 50
    })),
  clearHistory: () => set({ analysisHistory: [] }),
  toggleFaceRecognition: () => 
    set((state) => ({ faceRecognitionEnabled: !state.faceRecognitionEnabled })),
  setConfidenceThreshold: (threshold) => set({ confidenceThreshold: threshold }),
}));
