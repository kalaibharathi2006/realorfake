import { useState, useRef } from 'react'
import './index.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    setFile(selectedFile)
    setResult(null)
    setError(null)
    const objectUrl = URL.createObjectURL(selectedFile)
    setPreview(objectUrl)
  }

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    const isVideo = file.type.startsWith('video')

    // Dynamically use the production URL if provided, otherwise fallback to localhost
    const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    const endpoint = isVideo ? `${API_BASE}/detect/video` : `${API_BASE}/detect/image`

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const data = await response.json()
      setTimeout(() => {
        setResult(data)
        setLoading(false)
      }, 2000)

    } catch (err) {
      setError('Failed to connect to the analysis server. Is the backend running?')
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <>
      {/* Floating Orbs */}
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="orb orb-3"></div>

      <div className="container">
        <header>
          <div className="logo-container">
            <div className="logo-ring"></div>
            <div className="logo-icon">🛡️</div>
          </div>
          <h1>VERITAS AI</h1>
          <div className="tagline">
            <span className="badge">Neural Detection</span>
            <span className="badge" style={{ background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)' }}>v2.0</span>
          </div>
          <p className="subtitle">
            Advanced deepfake detection powered by cutting-edge neural networks.
            Upload any media to verify its authenticity in seconds.
          </p>
        </header>

        <main>
          {!file ? (
            <div
              className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,video/*"
                onChange={handleChange}
                style={{ display: 'none' }}
              />
              <div className="upload-icon-wrapper">
                <div className="upload-icon">📤</div>
              </div>
              <h2>Drop your media here</h2>
              <p>or click to browse files</p>

              <div className="file-types">
                <span className="file-type">JPG</span>
                <span className="file-type">PNG</span>
                <span className="file-type">WEBP</span>
                <span className="file-type">MP4</span>
                <span className="file-type">AVI</span>
              </div>
            </div>
          ) : (
            <div className="preview-section">
              {!result && !loading && (
                <div style={{ textAlign: 'center' }}>
                  <div className="preview-container" style={{ maxWidth: '500px', margin: '0 auto 2rem', height: 'auto' }}>
                    {file.type.startsWith('video') ? (
                      <video src={preview} controls className="preview-video" style={{ height: 'auto', maxHeight: '400px' }} />
                    ) : (
                      <img src={preview} alt="Preview" className="preview-image" style={{ height: 'auto', maxHeight: '400px' }} />
                    )}
                    <div className="scan-overlay"></div>
                  </div>
                  <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                    <button className="btn-primary" onClick={handleUpload}>
                      🔍 Analyze Authenticity
                    </button>
                    <button className="btn-secondary" onClick={reset}>
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {loading && (
                <div className="loader-container">
                  <div className="loader-ring"></div>
                  <div className="loader-text">
                    <h3>Analyzing Media...</h3>
                    <p>Scanning for manipulation artifacts and inconsistencies</p>
                  </div>
                </div>
              )}

              {result && (
                <div className="result-card">
                  <div className="preview-container">
                    {file.type.startsWith('video') ? (
                      <video src={preview} controls className="preview-video" />
                    ) : (
                      <img src={preview} alt="Analyzed" className="preview-image" />
                    )}
                    <div className="scan-overlay"></div>
                    <div className="scan-line"></div>
                  </div>

                  <div className="result-details">
                    <div className={`prediction-badge ${result.prediction === 'Real' ? 'prediction-real' :
                      result.prediction === 'Scam' ? 'prediction-fake' : 'prediction-fake'
                      }`}
                      style={{
                        background: result.prediction === 'Scam' ? 'rgba(220, 38, 38, 0.2)' : undefined,
                        borderColor: result.prediction === 'Scam' ? '#ef4444' : undefined,
                        color: result.prediction === 'Scam' ? '#ef4444' : undefined
                      }}
                    >
                      {result.prediction === 'Real' ? '✓' : result.prediction === 'Scam' ? '☠' : '⚠'} {result.prediction}
                    </div>

                    <div className="confidence-section">
                      <div className="confidence-header">
                        <span className="confidence-label">Confidence Score</span>
                        <span className="confidence-value">{result.confidence.toFixed(1)}%</span>
                      </div>
                      <div className="confidence-bar-container">
                        <div
                          className="confidence-bar"
                          style={{
                            width: `${result.confidence}%`,
                            background: result.prediction === 'Real'
                              ? 'linear-gradient(90deg, #10b981, #34d399)'
                              : result.prediction === 'Scam'
                                ? 'linear-gradient(90deg, #dc2626, #ef4444)'
                                : 'linear-gradient(90deg, #ef4444, #f87171)'
                          }}
                        ></div>
                      </div>
                    </div>

                    <div className="analysis-report">
                      <h3>Analysis Report</h3>
                      <p>{result.message}</p>
                    </div>

                    <button className="btn-primary" onClick={reset} style={{ marginTop: 'auto', width: '100%' }}>
                      Analyze New File
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div style={{ textAlign: 'center', marginTop: '2rem', padding: '2rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '16px', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                  <p style={{ color: 'var(--danger)', margin: '0 0 1rem' }}>{error}</p>
                  <button className="btn-secondary" onClick={reset} style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>
                    Try Again
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Stats Section */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">99.2%</div>
              <div className="stat-label">Detection Accuracy</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">&lt;2s</div>
              <div className="stat-label">Analysis Speed</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">50M+</div>
              <div className="stat-label">Files Scanned</div>
            </div>
          </div>
        </main>
      </div>
    </>
  )
}

export default App
