import { create } from 'zustand'
import {
  VideoFile,
  InferenceResult,
  ModelInfo,
  AnomalyDetection,
} from '../types'

interface AppState {
  video: VideoFile | null
  setVideo: (video: VideoFile | null) => void

  inferenceResult: InferenceResult | null
  setInferenceResult: (result: InferenceResult | null) => void

  modelInfo: ModelInfo | null
  setModelInfo: (info: ModelInfo | null) => void

  isProcessing: boolean
  setIsProcessing: (processing: boolean) => void

  selectedDetection: AnomalyDetection | null
  setSelectedDetection: (detection: AnomalyDetection | null) => void

  anomalyThreshold: number
  setAnomalyThreshold: (threshold: number) => void

  batchSize: number
  setBatchSize: (size: number) => void

  error: string | null
  setError: (error: string | null) => void

  success: string | null
  setSuccess: (message: string | null) => void
}

export const useAppStore = create<AppState>((set: any) => ({
  video: null,
  setVideo: (video: VideoFile | null) => set({ video }),

  inferenceResult: null,
  setInferenceResult: (result: InferenceResult | null) => set({ inferenceResult: result }),

  modelInfo: null,
  setModelInfo: (info: ModelInfo | null) => set({ modelInfo: info }),

  isProcessing: false,
  setIsProcessing: (processing: boolean) => set({ isProcessing: processing }),

  selectedDetection: null,
  setSelectedDetection: (detection: AnomalyDetection | null) => set({ selectedDetection: detection }),

  anomalyThreshold: 0.5,
  setAnomalyThreshold: (threshold: number) => set({ anomalyThreshold: threshold }),

  batchSize: 8,
  setBatchSize: (size: number) => set({ batchSize: size }),

  error: null,
  setError: (error: string | null) => set({ error }),

  success: null,
  setSuccess: (message: string | null) => set({ success: message }),
}))
