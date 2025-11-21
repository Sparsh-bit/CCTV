export interface VideoFile {
  file: File
  preview: string
  uploadProgress: number
  uploadedAt?: string
}

export interface Person {
  id: string
  x: number
  y: number
  width: number
  height: number
  confidence: number
}

export interface AnomalyDetection {
  frame_idx: number
  timestamp: number
  anomaly_score: number
  num_people: number
  people: Person[]
}

export interface InferenceResult {
  status: 'pending' | 'processing' | 'completed' | 'error'
  video_path: string
  frames_processed: number
  total_frames: number
  anomaly_scores: number[]
  heatmap_data: HeatmapData
  detections: AnomalyDetection[]
  processing_time_sec: number
  throughput_fps: number
  error_message?: string
}

export interface HeatmapData {
  width: number
  height: number
  frames: HeatmapFrame[]
}

export interface HeatmapFrame {
  frame_idx: number
  data: number[][]
}

export interface ModelInfo {
  model_type: string
  parameters: number
  input_dim: number
  d_model: number
  num_heads: number
  num_layers: number
  training_loss: number
  device: string
}

export interface APIResponse<T> {
  status: 'success' | 'error'
  data?: T
  message?: string
  error?: string
}

export interface UploadResponse {
  video_id: string
  filename: string
  upload_url: string
  file_size: number
}

export interface InferenceRequest {
  video_id: string
  model_type: string
  anomaly_threshold: number
  batch_size: number
  frame_interval: number
}
