# Frontend - Crowd Behavior Forecasting System

## 🚀 Industry-Level React + TypeScript Frontend

A modern, production-ready web application for real-time crowd behavior analysis, anomaly detection, and trajectory tracking.

## ✨ Features

- **Modern React 18** with Hooks and Functional Components
- **TypeScript** for type safety and better development experience
- **Vite** for ultra-fast development and optimized builds
- **Zustand** for lightweight state management
- **Styled Components** for scoped CSS and dynamic styling
- **Axios** with interceptors for API integration
- **React Dropzone** for file uploads
- **Recharts** for data visualization
- **Responsive Design** that works on all devices
- **Real-time Progress Tracking**
- **Video Upload & Processing**
- **Heatmap Generation**
- **Crowd Detection Visualization**

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── VideoUpload.tsx          # Video upload component
│   │   ├── InferenceControls.tsx    # Inference settings
│   │   └── Visualizers.tsx          # Heatmap & detection visualization
│   ├── services/
│   │   └── api.ts                   # API client with interceptors
│   ├── store/
│   │   └── index.ts                 # Zustand store
│   ├── styles/
│   │   └── index.ts                 # Styled components
│   ├── types/
│   │   └── index.ts                 # TypeScript interfaces
│   ├── App.tsx                      # Main application component
│   ├── main.tsx                     # Entry point
│   └── index.css                    # Global styles
├── index.html                       # HTML template
├── tsconfig.json                    # TypeScript configuration
├── vite.config.ts                   # Vite configuration
├── package.json                     # Dependencies
└── README.md                        # Documentation
```

## 🛠️ Installation

### Prerequisites
- Node.js 16+ or higher
- npm or yarn package manager
- Backend running on http://localhost:8000

### Step 1: Install Dependencies

```bash
cd frontend
npm install
# or
yarn install
```

### Step 2: Configure Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_ENV=development
```

### Step 3: Start Development Server

```bash
npm run dev
# or
yarn dev
```

The application will be available at `http://localhost:3000`

## 📦 Build for Production

```bash
npm run build
# or
yarn build
```

The production-ready files will be in the `dist` folder.

## 🔌 Backend Integration

### API Endpoints Used

#### Videos
```
POST   /api/videos/upload              # Upload video file
```

#### Inference
```
POST   /api/inference/run               # Run inference on video
GET    /api/inference/status/{video_id} # Get inference status
GET    /api/inference/heatmap/{video_id}          # Get heatmap image
GET    /api/inference/detection-video/{video_id}  # Get detection video
```

#### Model
```
GET    /api/model/info                 # Get model information
```

### API Client Usage

The `APIClient` class in `src/services/api.ts` handles all backend communication:

```typescript
import { apiClient } from '@services/api'

// Upload video
const uploadResponse = await apiClient.uploadVideo(file, (progress) => {
  console.log(`Upload progress: ${progress}%`)
})

// Run inference
const result = await apiClient.runInference(request, (progress) => {
  console.log(`Inference progress: ${progress}%`)
})

// Get heatmap
const heatmap = await apiClient.getHeatmap(videoId)

// Get detection video
const detectionVideo = await apiClient.getDetectionVideo(videoId)

// Get model info
const modelInfo = await apiClient.getModelInfo()
```

## 🎨 UI Components

### VideoUpload Component
Handles drag-and-drop video uploads with progress tracking.

```typescript
<VideoUpload
  onUpload={handleVideoUpload}
  isLoading={isProcessing}
/>
```

### InferenceControls Component
Settings for inference with threshold and batch size controls.

```typescript
<InferenceControls
  onRunInference={handleRunInference}
  onThresholdChange={setAnomalyThreshold}
  onBatchSizeChange={setBatchSize}
  isLoading={isProcessing}
/>
```

### Visualizers Components
Display heatmaps and crowd detection overlays on video.

```typescript
<HeatmapVisualizer
  imageData={heatmapData}
  width={1280}
  height={720}
  title="Anomaly Heatmap"
/>

<DetectionVisualizer
  videoUrl={videoUrl}
  detections={detections}
  title="Crowd Detection"
/>
```

## 📊 State Management

Using Zustand for global state management:

```typescript
import { useAppStore } from '@store'

// In component
const { video, setVideo, inferenceResult, isProcessing } = useAppStore()
```

## 🔄 Workflow

1. **Upload Video**: User uploads a video file (drag & drop or click)
2. **Configure Settings**: Set anomaly threshold and batch size
3. **Run Inference**: Start the inference process
4. **View Results**: 
   - See anomaly scores over time
   - View crowd detection with bounding boxes
   - Examine heatmaps showing anomaly regions
   - Analyze individual detections

## 🎯 Key Features Implemented

### ✅ Video Upload
- Drag & drop support
- File type validation
- Progress tracking
- File metadata display

### ✅ Inference Settings
- Adjustable anomaly threshold (0.0 - 1.0)
- Configurable batch size (1 - 64)
- Real-time parameter updates

### ✅ Results Visualization
- Frame-by-frame anomaly scores
- Crowd detection with bounding boxes
- Color-coded anomaly indicators (green=normal, red=anomaly)
- Heatmap visualization
- Processing statistics

### ✅ Error Handling
- Comprehensive error messages
- Retry logic
- Fallback UI states
- Loading indicators

## 🚀 Deployment

### Using Vercel (Recommended)

```bash
npm install -g vercel
vercel
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "preview"]
```

Build and run:

```bash
docker build -t crowd-forecasting-frontend .
docker run -p 3000:3000 crowd-forecasting-frontend
```

## 📱 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 🔐 Security

- CORS configured for frontend-backend communication
- Input validation for all uploads
- Type-safe with TypeScript
- Secure API headers with authorization support

## 📚 Technologies Used

| Technology | Purpose |
|-----------|---------|
| React 18 | UI Framework |
| TypeScript | Type Safety |
| Vite | Build Tool |
| Styled Components | CSS-in-JS |
| Zustand | State Management |
| Axios | HTTP Client |
| React Dropzone | File Upload |
| Recharts | Data Visualization |

## 🐛 Development Tips

### Enable Source Maps
Source maps are automatically enabled in development for easier debugging.

### TypeScript Checking
Run TypeScript compiler:
```bash
npm run lint
```

### Component Development
Components are organized by feature in `src/components/`.

### API Testing
Test API endpoints using Postman or curl:
```bash
curl -X POST http://localhost:8000/api/videos/upload \
  -F "file=@video.mp4"
```

## 🤝 Contributing

To contribute to the frontend:

1. Create a feature branch
2. Make your changes
3. Ensure TypeScript builds without errors
4. Test on multiple browsers
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 📞 Support

For issues or questions:
1. Check the [API Documentation](#backend-integration)
2. Review component PropTypes in TypeScript
3. Check console for error messages
4. Verify backend is running on port 8000

---

**Built with ❤️ for Crowd Behavior Forecasting**
