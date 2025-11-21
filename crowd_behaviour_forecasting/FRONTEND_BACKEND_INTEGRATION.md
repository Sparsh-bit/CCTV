# 🔗 Frontend-Backend Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND (React + TS)                 │
│                   http://localhost:3000                  │
├─────────────────────────────────────────────────────────┤
│  Components │ State │ API Client │ Visualizers           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST API
                     │ JSON over HTTP
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  BACKEND (FastAPI)                       │
│                  http://localhost:8000                   │
├─────────────────────────────────────────────────────────┤
│  Videos │ Inference │ Results │ Model API               │
├─────────────────────────────────────────────────────────┤
│  ML Pipeline: Video → Detection → Heatmap → Results     │
└─────────────────────────────────────────────────────────┘
```

## 📋 Setup Instructions

### Step 1: Backend Setup

```bash
# Navigate to project root
cd crowd_behaviour_forecasting

# Install backend dependencies
pip install -r requirements.txt

# Start FastAPI server
python -m uvicorn src.api.backend:app --host 0.0.0.0 --port 8000 --reload

# Backend running at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

### Step 2: Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Start development server
npm run dev

# Frontend running at: http://localhost:3000
```

## 🔄 Data Flow

### Video Upload Flow

```
User uploads video
        ↓
VideoUpload component captures file
        ↓
APIClient.uploadVideo() → POST /api/videos/upload
        ↓
Backend saves to uploads/ folder
        ↓
Returns video_id to frontend
        ↓
Store video_id in Zustand store
        ↓
Update UI with success message
```

### Inference Flow

```
User clicks "Run Inference"
        ↓
InferenceControls captures settings
        ↓
APIClient.runInference() → POST /api/inference/run
        ↓
Backend starts processing
        ↓
Processes video frames
        ↓
Runs Transformer model
        ↓
Generates detections & heatmap
        ↓
Returns InferenceResult JSON
        ↓
Frontend stores results in Zustand
        ↓
Visualizers render heatmap & detections
```

## 🔌 API Contracts

### Request/Response Examples

#### Upload Video

**Request:**
```typescript
const file: File = // user selected file
await apiClient.uploadVideo(file, (progress) => console.log(progress))
```

**Response:**
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "crowd.mp4",
  "file_size": 5242880,
  "upload_time": "2025-11-21T12:30:00.000Z"
}
```

#### Run Inference

**Request:**
```typescript
const request: InferenceRequest = {
  video_id: "550e8400-e29b-41d4-a716-446655440000",
  model_type: "transformer",
  anomaly_threshold: 0.5,
  batch_size: 8,
  frame_interval: 1
}
await apiClient.runInference(request)
```

**Response:**
```json
{
  "status": "completed",
  "video_path": "/path/to/video.mp4",
  "frames_processed": 30,
  "total_frames": 1800,
  "anomaly_scores": [0.45, 0.46, 0.44, ...],
  "detections": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "anomaly_score": 0.45,
      "num_people": 5,
      "people": [
        {
          "id": "person_0",
          "x": 100,
          "y": 150,
          "width": 80,
          "height": 120,
          "confidence": 0.95,
          "anomaly_score": 0.4
        },
        ...
      ]
    },
    ...
  ],
  "processing_time_sec": 2.5,
  "throughput_fps": 12.0
}
```

## 🛠️ Development Workflow

### Adding a New Component

1. Create component in `frontend/src/components/`
2. Define TypeScript interfaces in `frontend/src/types/`
3. Use Zustand store for global state if needed
4. Style using styled-components in `frontend/src/styles/`
5. Import and use in App.tsx

### Adding a New API Endpoint

1. Add endpoint to `src/api/backend.py`
2. Define Pydantic models for request/response
3. Add method to `frontend/src/services/api.ts`
4. Use in component with proper error handling

### Error Handling

Frontend automatically handles:
- Network errors
- Backend errors (401, 404, 500, etc.)
- Timeout errors
- File upload errors

Example:
```typescript
try {
  await apiClient.runInference(request)
} catch (error) {
  const message = error instanceof Error ? error.message : 'Unknown error'
  setError(message)
}
```

## 📊 State Management Pattern

Using Zustand for global state:

```typescript
// Define store
interface AppState {
  video: VideoFile | null
  setVideo: (video: VideoFile | null) => void
  // ... other state
}

export const useAppStore = create<AppState>((set) => ({
  video: null,
  setVideo: (video) => set({ video }),
  // ... other actions
}))

// Use in component
const { video, setVideo } = useAppStore()
```

## 🔐 Security Considerations

### CORS Configuration
Backend is configured to accept requests from frontend:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Authentication
Add JWT tokens if needed:
```typescript
// In APIClient
const token = localStorage.getItem('auth_token')
if (token) {
  config.headers.Authorization = `Bearer ${token}`
}
```

### Input Validation
All inputs are validated:
- File type checks in frontend
- Pydantic validation in backend
- TypeScript type checking

## 📈 Performance Optimization

### Frontend Optimizations
1. Code splitting with Vite
2. Tree-shaking unused code
3. Minification in production build
4. Image optimization
5. Lazy loading components

### Backend Optimizations
1. Model caching in memory
2. Batch processing for efficiency
3. Async/await for concurrent requests
4. Result caching

## 🧪 Testing

### Frontend Testing
```bash
# Run TypeScript check
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend Testing
```bash
# Run API with verbose logging
python -m uvicorn src.api.backend:app --log-level debug

# Test endpoints
curl -X GET http://localhost:8000/health
curl -X POST http://localhost:8000/api/videos/upload -F "file=@video.mp4"
```

## 📝 Troubleshooting

### Frontend won't connect to backend

**Problem**: `Cannot connect to http://localhost:8000`
**Solution**: 
1. Ensure backend is running: `python -m uvicorn src.api.backend:app --port 8000`
2. Check REACT_APP_API_URL in .env file
3. Verify CORS middleware is configured

### File upload fails

**Problem**: Upload stuck or 400 error
**Solution**:
1. Check file size (should be < 100MB)
2. Verify file format (MP4, AVI, MOV, MKV)
3. Check uploads/ folder exists
4. Check disk space

### Inference takes too long

**Problem**: Processing stuck after long time
**Solution**:
1. Check backend is running on CPU
2. Verify model is loaded correctly
3. Reduce batch_size in settings
4. Check logs for errors

## 📞 Debugging

### Frontend Console
Open browser DevTools (F12) → Console tab to see:
- API requests/responses
- Component state changes
- Error messages

### Backend Logs
Run with verbose logging:
```bash
python -m uvicorn src.api.backend:app --log-level debug
```

## 🚀 Deployment Checklist

- [ ] Build frontend: `npm run build`
- [ ] Test production build: `npm run preview`
- [ ] Update REACT_APP_API_URL to production backend
- [ ] Test all API endpoints
- [ ] Set up HTTPS
- [ ] Configure CORS for production
- [ ] Set up environment variables
- [ ] Deploy frontend to hosting (Vercel, Netlify, etc.)
- [ ] Deploy backend to server
- [ ] Monitor logs and errors

---

**For questions or issues, refer to respective READMEs:**
- Frontend: `frontend/README.md`
- Backend: `DEPLOYMENT_SUMMARY.md`
