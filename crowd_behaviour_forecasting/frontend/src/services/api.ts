import axios, { AxiosInstance, AxiosError, AxiosResponse } from 'axios'
import {
  APIResponse,
  UploadResponse,
  InferenceResult,
  ModelInfo,
  InferenceRequest,
} from '../types'

export class APIClient {
  private client: AxiosInstance

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    })

    this.setupInterceptors()
  }

  private setupInterceptors() {
    this.client.interceptors.request.use(
      (config: any) => {
        const token = localStorage.getItem('auth_token')
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error: any) => Promise.reject(error)
    )

    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: AxiosError<APIResponse<any>>) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token')
          window.location.href = '/login'
        }
        return Promise.reject(error)
      }
    )
  }

  async uploadVideo(file: File, onProgress?: (progress: number) => void): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await this.client.post<APIResponse<UploadResponse>>('/videos/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent: any) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress?.(percentCompleted)
          }
        },
      } as any)

      const apiResponse = response.data as APIResponse<UploadResponse>
      if (apiResponse.status === 'success' && apiResponse.data) {
        return apiResponse.data
      }
      throw new Error('Upload failed')
    } catch (error) {
      throw this.handleError(error)
    }
  }

  async runInference(request: InferenceRequest, onProgress?: (progress: number) => void): Promise<InferenceResult> {
    try {
      const response = await this.client.post<APIResponse<InferenceResult>>('/inference/run', request, {
        onDownloadProgress: (progressEvent: any) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress?.(percentCompleted)
          }
        },
      } as any)

      const apiResponse = response.data as APIResponse<InferenceResult>
      if (apiResponse.status === 'success' && apiResponse.data) {
        return apiResponse.data
      }
      throw new Error('Inference failed')
    } catch (error) {
      throw this.handleError(error)
    }
  }

  async getInferenceStatus(videoId: string): Promise<InferenceResult> {
    try {
      const response = await this.client.get<APIResponse<InferenceResult>>(`/inference/status/${videoId}`)
      const apiResponse = response.data as APIResponse<InferenceResult>
      if (apiResponse.status === 'success' && apiResponse.data) {
        return apiResponse.data
      }
      throw new Error('Failed to get status')
    } catch (error) {
      throw this.handleError(error)
    }
  }

  async getModelInfo(): Promise<ModelInfo> {
    try {
      const response = await this.client.get<APIResponse<ModelInfo>>('/model/info')
      const apiResponse = response.data as APIResponse<ModelInfo>
      if (apiResponse.status === 'success' && apiResponse.data) {
        return apiResponse.data
      }
      throw new Error('Failed to get model info')
    } catch (error) {
      throw this.handleError(error)
    }
  }

  async getHeatmap(videoId: string): Promise<Blob> {
    try {
      const response = await this.client.get(`/inference/heatmap/${videoId}`, { responseType: 'blob' })
      return response.data
    } catch (error) {
      throw this.handleError(error)
    }
  }

  async getDetectionVideo(videoId: string): Promise<Blob> {
    try {
      const response = await this.client.get(`/inference/detection-video/${videoId}`, { responseType: 'blob' })
      return response.data
    } catch (error) {
      throw this.handleError(error)
    }
  }

  private handleError(error: any): Error {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.error || error.response?.data?.message || error.message
      return new Error(message)
    }
    return error instanceof Error ? error : new Error('Unknown error occurred')
  }
}

const BASE_URL = '/api'
export const apiClient = new APIClient(BASE_URL)
