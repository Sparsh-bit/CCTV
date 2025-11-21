import styled from 'styled-components'

export const Container = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
`

export const Header = styled.header`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px;
  border-radius: 10px;
  margin-bottom: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);

  h1 {
    margin: 0 0 10px 0;
    font-size: 2.5rem;
    font-weight: 700;
  }

  p {
    margin: 0;
    font-size: 1.1rem;
    opacity: 0.9;
  }
`

export const TabContainer = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  flex-wrap: wrap;
`

export const TabButton = styled.button<{ active?: boolean }>`
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  background: ${(props) => (props.active ? '#667eea' : 'white')};
  color: ${(props) => (props.active ? 'white' : '#333')};
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
`

export const Card = styled.div`
  background: white;
  border-radius: 10px;
  padding: 30px;
  margin-bottom: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;

  &:hover {
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
  }

  h2 {
    margin: 0 0 20px 0;
    color: #333;
    font-size: 1.8rem;
    border-bottom: 3px solid #667eea;
    padding-bottom: 10px;
  }

  h3 {
    margin: 0 0 15px 0;
    color: #555;
    font-size: 1.3rem;
  }
`

export const Grid = styled.div<{ columns?: number }>`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
`

export const StatCard = styled(Card)`
  text-align: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;

  h3 {
    color: white;
    opacity: 0.9;
    border: none;
    margin-bottom: 10px;
  }

  .stat-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 15px 0;
  }

  .stat-label {
    margin: 0;
    opacity: 0.9;
    font-size: 1rem;
  }
`

export const InputGroup = styled.div`
  margin-bottom: 20px;

  label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #333;
    font-size: 1rem;
  }

  input,
  select,
  textarea {
    width: 100%;
    padding: 12px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 1rem;
    font-family: inherit;
    transition: all 0.3s ease;

    &:focus {
      outline: none;
      border-color: #667eea;
      box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    &:disabled {
      background: #f5f5f5;
      cursor: not-allowed;
    }
  }

  input[type='range'] {
    width: calc(100% - 60px);
    margin-right: 10px;
  }

  .range-value {
    display: inline-block;
    min-width: 50px;
    text-align: right;
    font-weight: 600;
    color: #667eea;
  }
`

export const ButtonGroup = styled.div`
  display: flex;
  gap: 10px;
  margin-top: 20px;
  flex-wrap: wrap;
`

export const Button = styled.button<{ variant?: 'primary' | 'secondary' | 'danger' }>`
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  flex: 1;
  min-width: 120px;

  background: ${(props) => {
    switch (props.variant) {
      case 'secondary':
        return '#6c757d'
      case 'danger':
        return '#dc3545'
      default:
        return '#667eea'
    }
  }};
  color: white;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`

export const DropZone = styled.div<{ isDragActive?: boolean }>`
  border: 3px dashed ${(props) => (props.isDragActive ? '#667eea' : '#d0d0d0')};
  border-radius: 10px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: ${(props) => (props.isDragActive ? '#f0f4ff' : '#f9f9f9')};

  p {
    margin: 10px 0;
    color: #666;
    font-size: 1.1rem;
  }

  .upload-icon {
    font-size: 3rem;
    margin-bottom: 10px;
  }
`

export const ProgressBar = styled.div<{ progress: number }>`
  width: 100%;
  height: 8px;
  background: #e0e0e0;
  border-radius: 10px;
  overflow: hidden;
  margin: 15px 0;

  &::after {
    content: '';
    display: block;
    height: 100%;
    width: ${(props) => props.progress}%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    transition: width 0.3s ease;
  }
`

export const Alert = styled.div<{ type?: 'success' | 'error' | 'info' }>`
  padding: 15px 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  border-left: 4px solid
    ${(props) => {
      switch (props.type) {
        case 'success':
          return '#28a745'
        case 'error':
          return '#dc3545'
        default:
          return '#17a2b8'
      }
    }};
  background: ${(props) => {
    switch (props.type) {
      case 'success':
        return '#d4edda'
      case 'error':
        return '#f8d7da'
      default:
        return '#d1ecf1'
    }
  }};
  color: ${(props) => {
    switch (props.type) {
      case 'success':
        return '#155724'
      case 'error':
        return '#721c24'
      default:
        return '#0c5460'
    }
  }};
`

export const VideoContainer = styled.div`
  position: relative;
  width: 100%;
  max-width: 100%;
  margin: 20px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);

  video {
    width: 100%;
    height: auto;
    display: block;
  }

  canvas {
    position: absolute;
    top: 0;
    left: 0;
  }
`

export const ChartContainer = styled.div`
  width: 100%;
  height: 400px;
  margin: 20px 0;
  padding: 20px;
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`

export const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;

  th,
  td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
  }

  th {
    background: #f5f5f5;
    font-weight: 600;
    color: #333;
  }

  tbody tr:hover {
    background: #f9f9f9;
  }
`

export const HeatmapContainer = styled.div`
  position: relative;
  width: 100%;
  max-width: 100%;
  margin: 20px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);

  canvas {
    width: 100%;
    height: auto;
    display: block;
  }
`

export const Loading = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;

  .spinner {
    width: 50px;
    height: 50px;
    border: 4px solid #f0f0f0;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  p {
    margin-top: 15px;
    color: #666;
    font-size: 1.1rem;
  }
`

export const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
  margin: 20px 0;

  .info-item {
    display: flex;
    justify-content: space-between;
    padding: 15px;
    background: #f9f9f9;
    border-radius: 8px;
    border-left: 4px solid #667eea;

    .label {
      font-weight: 600;
      color: #555;
    }

    .value {
      color: #667eea;
      font-weight: 700;
    }
  }
`

export const Footer = styled.footer`
  text-align: center;
  padding: 30px;
  color: white;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
  margin-top: 50px;

  p {
    margin: 5px 0;
    font-size: 0.95rem;
  }
`
