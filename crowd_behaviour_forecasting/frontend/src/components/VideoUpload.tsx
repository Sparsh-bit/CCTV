import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import * as S from '@/styles'

interface VideoUploadProps {
  onUpload: (file: File) => Promise<void>
  isLoading?: boolean
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  onUpload,
  isLoading = false,
}) => {
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        await onUpload(acceptedFiles[0])
      }
    },
    [onUpload]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv'] },
    multiple: false,
    disabled: isLoading,
  })

  return (
    <S.Card>
      <h2>Upload Video</h2>
      <S.DropZone {...getRootProps()} isDragActive={isDragActive}>
        <input {...getInputProps()} />
        <div className="upload-icon">📹</div>
        {isDragActive ? (
          <p>Drop the video file here...</p>
        ) : (
          <>
            <p>Drag & drop a video file here, or click to select</p>
            <p style={{ fontSize: '0.9rem', color: '#999' }}>
              Supported formats: MP4, AVI, MOV, MKV
            </p>
          </>
        )}
      </S.DropZone>
      {isLoading && (
        <S.Loading>
          <div className="spinner"></div>
          <p>Uploading video...</p>
        </S.Loading>
      )}
    </S.Card>
  )
}
