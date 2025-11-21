import React, { useState } from 'react'
import * as S from '@/styles'

interface InferenceControlsProps {
  onRunInference: () => Promise<void>
  onThresholdChange: (threshold: number) => void
  onBatchSizeChange: (size: number) => void
  isLoading?: boolean
  anomalyThreshold?: number
  batchSize?: number
}

export const InferenceControls: React.FC<InferenceControlsProps> = ({
  onRunInference,
  onThresholdChange,
  onBatchSizeChange,
  isLoading = false,
  anomalyThreshold = 0.5,
  batchSize = 8,
}) => {
  const [localThreshold, setLocalThreshold] = useState(anomalyThreshold)
  const [localBatchSize, setLocalBatchSize] = useState(batchSize)

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value)
    setLocalThreshold(value)
    onThresholdChange(value)
  }

  const handleBatchSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10)
    setLocalBatchSize(value)
    onBatchSizeChange(value)
  }

  return (
    <S.Card>
      <h2>Inference Settings</h2>

      <S.InputGroup>
        <label htmlFor="threshold">Anomaly Threshold</label>
        <div>
          <input
            id="threshold"
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={localThreshold}
            onChange={handleThresholdChange}
            disabled={isLoading}
          />
          <span className="range-value">{localThreshold.toFixed(1)}</span>
        </div>
        <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '5px' }}>
          Anomaly scores above this threshold will be highlighted
        </p>
      </S.InputGroup>

      <S.InputGroup>
        <label htmlFor="batchSize">Batch Size</label>
        <input
          id="batchSize"
          type="number"
          min="1"
          max="64"
          value={localBatchSize}
          onChange={handleBatchSizeChange}
          disabled={isLoading}
        />
        <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '5px' }}>
          Number of frames to process at once
        </p>
      </S.InputGroup>

      <S.ButtonGroup>
        <S.Button
          variant="primary"
          onClick={onRunInference}
          disabled={isLoading}
        >
          {isLoading ? 'Processing...' : 'Run Inference'}
        </S.Button>
      </S.ButtonGroup>
    </S.Card>
  )
}
