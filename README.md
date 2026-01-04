# 🛡️ VERITAS AI - Fake News Detector

A deep learning-powered platform for detecting fake news, deepfakes, and scam images.

![Platform](https://img.shields.io/badge/Platform-Web-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![React](https://img.shields.io/badge/React-18+-61DAFB)

## Features

- 🖼️ **Image Analysis** - Detect manipulated or fake images
- 🎬 **Video Analysis** - Analyze videos for deepfake patterns
- ⚠️ **Scam Detection** - Match against known scam patterns
- 🎨 **Modern UI** - Beautiful, responsive interface

## Quick Start

### 1. Start Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Backend runs at: http://localhost:8000

### 2. Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: http://localhost:5173

## Deployment (Production)

When deploying the frontend (e.g., to Vercel), set the following Environment Variable so it can talk to your backend:

- `VITE_API_URL`: Your backend URL (e.g., `https://your-backend.onrender.com`)

## Project Structure

```
fake_news_platform/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── model/
│   │   ├── architecture.py  # Neural network model
│   │   ├── inference.py     # Prediction logic
│   │   ├── scam_matcher.py  # Scam pattern matching
│   │   ├── dataset.py       # Data loading
│   │   └── train.py         # Model training
│   └── data/
│       └── scam_patterns/   # Known scam images
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   └── index.css        # Styling
│   └── package.json
└── README.md
```

## Adding Scam Patterns

Place known scam images in `backend/data/scam_patterns/` and restart the backend.

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: React, Vite
- **ML**: CNN-based deepfake detection

## License

MIT License
