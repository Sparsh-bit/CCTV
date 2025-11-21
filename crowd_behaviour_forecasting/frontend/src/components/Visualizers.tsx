import React, { useEffect, useRef } from 'react'
import * as S from '@/styles'

interface HeatmapVisualizerProps {
  imageData?: ImageData
  width?: number
  height?: number
  title?: string
}

export const HeatmapVisualizer: React.FC<HeatmapVisualizerProps> = ({
  imageData,
  width = 1280,
  height = 720,
  title = 'Anomaly Heatmap',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || !imageData) return

    const ctx = canvasRef.current.getContext('2d')
    if (ctx) {
      ctx.putImageData(imageData, 0, 0)
    }
  }, [imageData])

  return (
    <S.Card>
      <h3>{title}</h3>
      <S.HeatmapContainer>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ width: '100%', height: 'auto' }}
        />
      </S.HeatmapContainer>
    </S.Card>
  )
}

interface DetectionVisualizerProps {
  videoUrl?: string
  canvasWidth?: number
  canvasHeight?: number
  detections?: Array<{
    x: number
    y: number
    width: number
    height: number
    confidence: number
    anomaly_score?: number
  }>
  title?: string
}

export const DetectionVisualizer: React.FC<DetectionVisualizerProps> = ({
  videoUrl,
  canvasWidth = 1280,
  canvasHeight = 720,
  detections = [],
  title = 'Crowd Detection',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const video = videoRef.current

    const drawFrame = () => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      detections.forEach((detection) => {
        const isAnomaly = detection.anomaly_score !== undefined && detection.anomaly_score > 0.5

        ctx.strokeStyle = isAnomaly ? '#ff4444' : '#00cc00'
        ctx.lineWidth = 3
        ctx.strokeRect(
          detection.x,
          detection.y,
          detection.width,
          detection.height
        )

        ctx.fillStyle = isAnomaly ? '#ff4444' : '#00cc00'
        ctx.font = '12px Arial'
        ctx.fillText(
          `${(detection.confidence * 100).toFixed(0)}%`,
          detection.x,
          detection.y - 5
        )

        if (isAnomaly) {
          ctx.fillStyle = 'rgba(255, 0, 0, 0.2)'
          ctx.fillRect(
            detection.x,
            detection.y,
            detection.width,
            detection.height
          )
        }
      })

      requestAnimationFrame(drawFrame)
    }

    if (video.readyState === video.HAVE_ENOUGH_DATA) {
      drawFrame()
    } else {
      video.onloadedmetadata = () => {
        drawFrame()
      }
    }
  }, [detections])

  return (
    <S.Card>
      <h3>{title}</h3>
      <S.VideoContainer>
        <video
          ref={videoRef}
          src={videoUrl}
          width={canvasWidth}
          height={canvasHeight}
          controls
        />
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
          }}
        />
      </S.VideoContainer>
    </S.Card>
  )
}
