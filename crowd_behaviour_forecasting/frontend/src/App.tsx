import React, { useState } from 'react'
import * as S from '@/styles'
import { VideoUpload } from '@components/VideoUpload'
import { InferenceControls } from '@components/InferenceControls'
import { HeatmapVisualizer, DetectionVisualizer } from '@components/Visualizers'
import { apiClient } from '@services/api'
import { useAppStore } from '@store/index'

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('upload')
  const {
    video,
    setVideo,
    inferenceResult,
    setInferenceResult,
    isProcessing,
    setIsProcessing,
    anomalyThreshold,
    setAnomalyThreshold,
    batchSize,
    setBatchSize,
    error,
    setError,
    success,
    setSuccess,
  } = useAppStore()

  const [uploadProgress, setUploadProgress] = useState(0)
  const [inferenceProgress, setInferenceProgress] = useState(0)

  const handleVideoUpload = async (file: File) => {
    try {
      setIsProcessing(true)
      setError(null)

      await apiClient.uploadVideo(file, (progress) => {
        setUploadProgress(progress)
      })

      const preview = URL.createObjectURL(file)
      setVideo({
        file,
        preview,
        uploadProgress: 100,
        uploadedAt: new Date().toISOString(),
      })

      setSuccess('Video uploaded successfully!')
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed'
      setError(errorMessage)
    } finally {
      setIsProcessing(false)
      setUploadProgress(0)
    }
  }

  const handleRunInference = async () => {
    if (!video) {
      setError('Please upload a video first')
      return
    }

    try {
      setIsProcessing(true)
      setError(null)
      setActiveTab('results')

      const request = {
        video_id: video.uploadedAt || 'temp',
        model_type: 'transformer',
        anomaly_threshold: anomalyThreshold,
        batch_size: batchSize,
        frame_interval: 1,
      }

      const result = await apiClient.runInference(request, (progress) => {
        setInferenceProgress(progress)
      })

      setInferenceResult(result)
      setSuccess('Inference completed successfully!')
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Inference failed'
      setError(errorMessage)
    } finally {
      setIsProcessing(false)
      setInferenceProgress(0)
    }
  }

  return (
    <S.Container>
      <S.Header>
        <h1>👥 Crowd Behavior Forecasting System</h1>
        <p>Real-time Anomaly Detection & Trajectory Analysis</p>
      </S.Header>

      {error && <S.Alert type="error">{error}</S.Alert>}
      {success && <S.Alert type="success">{success}</S.Alert>}

      <S.TabContainer>
        <S.TabButton
          active={activeTab === 'upload'}
          onClick={() => setActiveTab('upload')}
        >
          📹 Upload Video
        </S.TabButton>
        <S.TabButton
          active={activeTab === 'settings'}
          onClick={() => setActiveTab('settings')}
        >
          ⚙️ Settings
        </S.TabButton>
        <S.TabButton
          active={activeTab === 'results'}
          onClick={() => setActiveTab('results')}
        >
          📊 Results
        </S.TabButton>
      </S.TabContainer>

      {activeTab === 'upload' && (
        <>
          <VideoUpload
            onUpload={handleVideoUpload}
            isLoading={isProcessing}
          />
          {isProcessing && uploadProgress > 0 && uploadProgress < 100 && (
            <S.Card>
              <p>Uploading: {uploadProgress}%</p>
              <S.ProgressBar progress={uploadProgress} />
            </S.Card>
          )}
          {video && (
            <S.Card>
              <h3>Video Preview</h3>
              <video
                src={video.preview}
                width="100%"
                height="auto"
                controls
                style={{ borderRadius: '8px', marginTop: '10px' }}
              />
              <S.InfoGrid>
                <div className="info-item">
                  <span className="label">File Name:</span>
                  <span className="value">{video.file.name}</span>
                </div>
                <div className="info-item">
                  <span className="label">File Size:</span>
                  <span className="value">
                    {(video.file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
                <div className="info-item">
                  <span className="label">Upload Time:</span>
                  <span className="value">
                    {video.uploadedAt
                      ? new Date(video.uploadedAt).toLocaleString()
                      : 'N/A'}
                  </span>
                </div>
              </S.InfoGrid>
            </S.Card>
          )}
        </>
      )}

      {activeTab === 'settings' && (
        <InferenceControls
          onRunInference={handleRunInference}
          onThresholdChange={setAnomalyThreshold}
          onBatchSizeChange={setBatchSize}
          isLoading={isProcessing}
          anomalyThreshold={anomalyThreshold}
          batchSize={batchSize}
        />
      )}

      {activeTab === 'results' && (
        <>
          {isProcessing ? (
            <S.Loading>
              <div className="spinner"></div>
              <p>Processing inference... {inferenceProgress}%</p>
              <S.ProgressBar progress={inferenceProgress} />
            </S.Loading>
          ) : inferenceResult ? (
            <>
              <S.Card>
                <h2>Inference Results</h2>
                <S.InfoGrid>
                  <div className="info-item">
                    <span className="label">Status:</span>
                    <span className="value">{inferenceResult.status}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Frames Processed:</span>
                    <span className="value">
                      {inferenceResult.frames_processed}/
                      {inferenceResult.total_frames}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="label">Processing Time:</span>
                    <span className="value">
                      {inferenceResult.processing_time_sec.toFixed(2)}s
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="label">Throughput:</span>
                    <span className="value">
                      {inferenceResult.throughput_fps.toFixed(2)} FPS
                    </span>
                  </div>
                </S.InfoGrid>
              </S.Card>

              {video && (
                <>
                  <DetectionVisualizer
                    videoUrl={video.preview}
                    detections={inferenceResult.detections[0]?.people || []}
                    title="Crowd Detection with Anomalies"
                  />
                  <HeatmapVisualizer
                    title="Anomaly Heatmap"
                  />
                </>
              )}
            </>
          ) : (
            <S.Card>
              <p>No results yet. Upload a video and run inference to see results.</p>
            </S.Card>
          )}
        </>
      )}

      <S.Footer>
        <p>&copy; 2025 Crowd Behavior Forecasting System</p>
        <p>Status: OPERATIONAL | Model: Transformer | Device: CPU</p>
      </S.Footer>
    </S.Container>
  )
}

export default App
