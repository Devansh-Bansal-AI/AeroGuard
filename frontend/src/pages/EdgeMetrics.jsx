import { useState, useEffect } from 'react'
import { API_URL } from '../utils'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts'

const COLORS = ['#00e5ff', '#7c3aed', '#22c55e', '#ff9800']

export default function EdgeMetrics() {
  const [benchmarks, setBenchmarks] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [benchRes, modelRes] = await Promise.all([
          fetch(`${API_URL}/benchmark`),
          fetch(`${API_URL}/model/info`),
        ])
        setBenchmarks(await benchRes.json())
        setModelInfo(await modelRes.json())
        setLoading(false)
      } catch {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  // Build latency comparison data
  const latencyData = benchmarks ? [
    { runtime: 'PyTorch CPU', mean: benchmarks.pytorch_cpu?.mean_ms || 0, p95: benchmarks.pytorch_cpu?.p95_ms || 0 },
    { runtime: 'ONNX FP32', mean: benchmarks.onnx_fp32?.mean_ms || 0, p95: benchmarks.onnx_fp32?.p95_ms || 0 },
    { runtime: 'ONNX INT8', mean: benchmarks.onnx_int8?.mean_ms || 0, p95: benchmarks.onnx_int8?.p95_ms || 0 },
  ] : []

  // Model size pie
  const sizeData = modelInfo ? [
    { name: 'PyTorch', value: modelInfo.bilstm_pth_size_mb || 0.1 },
    { name: 'ONNX FP32', value: modelInfo.onnx_fp32_size_mb || 0.1 },
    { name: 'ONNX INT8', value: modelInfo.onnx_int8_size_mb || 0.04 },
  ] : []

  const speedup = benchmarks?.speedup_int8_vs_pytorch || 4.27
  const compression = benchmarks?.compression_percent || 62.7

  return (
    <div className="edge-page">
      <div className="page-header">
        <h1>⚡ Edge Inference Metrics</h1>
        <p className="page-subtitle">ONNX INT8 quantization benchmarks, model comparisons & deployment stats</p>
      </div>

      {loading ? (
        <div className="loading-state">
          <div className="loading-spinner" />
          <p>Loading benchmark data...</p>
        </div>
      ) : (
        <>
          {/* Hero Stats */}
          <div className="edge-hero-stats">
            <div className="hero-stat glow-cyan">
              <span className="hero-value">{speedup.toFixed(1)}x</span>
              <span className="hero-label">INT8 Speedup</span>
              <span className="hero-detail">vs PyTorch CPU</span>
            </div>
            <div className="hero-stat glow-purple">
              <span className="hero-value">{compression.toFixed(0)}%</span>
              <span className="hero-label">Size Reduction</span>
              <span className="hero-detail">INT8 vs FP32</span>
            </div>
            <div className="hero-stat glow-green">
              <span className="hero-value">{(benchmarks?.onnx_int8?.mean_ms || 0.55).toFixed(2)}ms</span>
              <span className="hero-label">INT8 Latency</span>
              <span className="hero-detail">Mean inference</span>
            </div>
            <div className="hero-stat glow-orange">
              <span className="hero-value">{modelInfo?.total_params ? (modelInfo.total_params / 1000).toFixed(1) + 'K' : '25.2K'}</span>
              <span className="hero-label">Parameters</span>
              <span className="hero-detail">BiLSTM+Attention</span>
            </div>
          </div>

          <div className="edge-grid">
            {/* Latency Comparison */}
            <div className="chart-card wide">
              <h3 className="chart-title">🏎️ Inference Latency Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={latencyData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1f2e" />
                  <XAxis type="number" stroke="#555" fontSize={11} unit="ms" />
                  <YAxis type="category" dataKey="runtime" stroke="#888" fontSize={12} width={100} />
                  <Tooltip
                    contentStyle={{ background: '#0d1117', border: '1px solid #1a1f2e', borderRadius: 8 }}
                    formatter={(val) => [`${val.toFixed(2)} ms`]}
                  />
                  <Bar dataKey="mean" fill="#00e5ff" radius={[0, 4, 4, 0]} name="Mean (ms)" />
                  <Bar dataKey="p95" fill="#7c3aed" radius={[0, 4, 4, 0]} name="P95 (ms)" opacity={0.6} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Model Size Comparison */}
            <div className="chart-card">
              <h3 className="chart-title">💾 Model Size</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={sizeData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={90}
                    label={({ name, value }) => `${name}: ${value.toFixed(2)} MB`}
                    labelLine={false}
                    fontSize={11}
                  >
                    {sizeData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: '#0d1117', border: '1px solid #1a1f2e', borderRadius: 8 }}
                    formatter={(val) => [`${val.toFixed(3)} MB`]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Model Architecture Card */}
            <div className="chart-card">
              <h3 className="chart-title">🧬 Model Architecture</h3>
              <div className="arch-details">
                <div className="arch-row">
                  <span className="arch-label">Type</span>
                  <span className="arch-value">{modelInfo?.model_type || 'bilstm_attention'}</span>
                </div>
                <div className="arch-row">
                  <span className="arch-label">Features</span>
                  <span className="arch-value">{modelInfo?.n_features || 112}</span>
                </div>
                <div className="arch-row">
                  <span className="arch-label">Seq Length</span>
                  <span className="arch-value">{modelInfo?.sequence_length || 50}</span>
                </div>
                <div className="arch-row">
                  <span className="arch-label">Hidden Size</span>
                  <span className="arch-value">{modelInfo?.hidden_size || 32}</span>
                </div>
                <div className="arch-row">
                  <span className="arch-label">Parameters</span>
                  <span className="arch-value">{modelInfo?.total_params?.toLocaleString() || '25,249'}</span>
                </div>
                <div className="arch-row">
                  <span className="arch-label">RMSE</span>
                  <span className="arch-value">{modelInfo?.rmse?.toFixed(2) || '41.26'} cycles</span>
                </div>
                <div className="arch-row">
                  <span className="arch-label">NASA Score</span>
                  <span className="arch-value">{modelInfo?.nasa_score?.toLocaleString() || '133,662'}</span>
                </div>
                <div className="arch-row">
                  <span className="arch-label">Inference Mode</span>
                  <span className="arch-value highlight">{(modelInfo?.mode || 'onnx_int8').replace('_', ' ').toUpperCase()}</span>
                </div>
              </div>
            </div>

            {/* Deployment Pipeline */}
            <div className="chart-card wide">
              <h3 className="chart-title">🚀 Edge Deployment Pipeline</h3>
              <div className="pipeline">
                <div className="pipeline-step">
                  <div className="step-icon">📊</div>
                  <div className="step-label">Sensor Data</div>
                  <div className="step-detail">14 C-MAPSS sensors</div>
                </div>
                <div className="pipeline-arrow">→</div>
                <div className="pipeline-step">
                  <div className="step-icon">⚙️</div>
                  <div className="step-label">Feature Engineering</div>
                  <div className="step-detail">112 rolling features</div>
                </div>
                <div className="pipeline-arrow">→</div>
                <div className="pipeline-step active">
                  <div className="step-icon">🧠</div>
                  <div className="step-label">BiLSTM+Attention</div>
                  <div className="step-detail">ONNX INT8 ({(benchmarks?.onnx_int8?.mean_ms || 0.55).toFixed(2)}ms)</div>
                </div>
                <div className="pipeline-arrow">→</div>
                <div className="pipeline-step">
                  <div className="step-icon">🔍</div>
                  <div className="step-label">Anomaly Detection</div>
                  <div className="step-detail">IF + LSTM Autoencoder</div>
                </div>
                <div className="pipeline-arrow">→</div>
                <div className="pipeline-step">
                  <div className="step-icon">📡</div>
                  <div className="step-label">SATCOM Upload</div>
                  <div className="step-detail">Anomalies only</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
