import { useState, useEffect, useRef, useCallback, Component } from 'react'
import './index.css'

// ─── Error Boundary ───────────────────────────────────────────────────

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '2rem', color: '#ff4d4d', background: '#0a0e1a', minHeight: '100vh', fontFamily: 'monospace' }}>
          <h1>AeroGuard — System Error</h1>
          <p>The dashboard encountered an error and has been safely halted.</p>
          <pre style={{ color: '#888', marginTop: '1rem' }}>{this.state.error?.message}</pre>
          <button onClick={() => this.setState({ hasError: false })} style={{ marginTop: '1rem', padding: '8px 16px', background: '#00e5ff', color: '#000', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
            Restart Dashboard
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

// ─── Configuration ────────────────────────────────────────────────────
const SSE_URL = import.meta.env.VITE_SSE_URL || 'http://localhost:8000/ws/stream'
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// ─── Utility Functions ────────────────────────────────────────────────

function getHealthColor(status) {
  switch (status) {
    case 'healthy': return 'var(--color-healthy)'
    case 'warning': return 'var(--color-warning)'
    case 'critical': return 'var(--color-critical)'
    default: return 'var(--color-text-dim)'
  }
}

function getHealthIcon(status) {
  switch (status) {
    case 'healthy': return '●'
    case 'warning': return '◆'
    case 'critical': return '▲'
    default: return '○'
  }
}

function formatRUL(rul) {
  return Math.round(rul)
}

// ─── Gauge Component ──────────────────────────────────────────────────

function RadialGauge({ value, max, label, unit, color, size = 120 }) {
  const percentage = Math.min((value / max) * 100, 100)
  const radius = (size - 16) / 2
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference * (1 - percentage / 100)

  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`}>
        {/* Background arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--color-surface-alt)"
          strokeWidth="6"
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * 0.25}
        />
        {/* Value arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset + circumference * 0.25}
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
          className="gauge-arc"
        />
      </svg>
      <div className="gauge-value">
        <span className="gauge-number" style={{ color }}>{typeof value === 'number' ? value.toFixed(1) : value}</span>
        <span className="gauge-unit">{unit}</span>
      </div>
      <span className="gauge-label">{label}</span>
    </div>
  )
}

// ─── Sparkline Component ──────────────────────────────────────────────

function Sparkline({ data, color = 'var(--color-accent)', width = 120, height = 32 }) {
  if (!data || data.length < 2) return <div style={{ width, height }} />

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height - ((val - min) / range) * (height - 4) - 2
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={width} height={height} className="sparkline">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  )
}

// ─── Engine Card Component ────────────────────────────────────────────

function EngineCard({ engine, isSelected, onClick }) {
  const healthColor = getHealthColor(engine.health_status)
  const healthIcon = getHealthIcon(engine.health_status)

  return (
    <div
      className={`engine-card ${isSelected ? 'selected' : ''} ${engine.health_status}`}
      onClick={onClick}
      id={`engine-card-${engine.engine_id}`}
    >
      <div className="engine-card-header">
        <div className="engine-name-row">
          <span className="health-indicator" style={{ color: healthColor }}>
            {healthIcon}
          </span>
          <h3>{engine.name || `Engine #${engine.engine_id}`}</h3>
        </div>
        <span className="health-badge" style={{
          background: healthColor + '22',
          color: healthColor,
          border: `1px solid ${healthColor}44`
        }}>
          {engine.health_status.toUpperCase()}
        </span>
      </div>

      <div className="engine-card-metrics">
        <div className="metric">
          <span className="metric-label">RUL</span>
          <span className="metric-value" style={{ color: engine.rul_prediction < 30 ? 'var(--color-critical)' : 'var(--color-text)' }}>
            {formatRUL(engine.rul_prediction)}
            <span className="metric-unit">cycles</span>
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Anomaly</span>
          <span className="metric-value" style={{ color: engine.anomaly_score > 0.5 ? 'var(--color-warning)' : 'var(--color-text-dim)' }}>
            {(engine.anomaly_score * 100).toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Cycle</span>
          <span className="metric-value">{engine.current_cycle}</span>
        </div>
        {engine.model_used && engine.model_used !== 'simulated' && (
          <div className="metric model-badge">
            <span className="metric-label">Inference</span>
            <span className="metric-value" style={{ color: 'var(--color-accent)', fontSize: '0.7rem' }}>
              XGBoost
            </span>
          </div>
        )}
      </div>

      {/* Life progress bar */}
      <div className="life-progress">
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{
              width: `${engine.life_progress}%`,
              background: engine.life_progress > 85
                ? 'var(--color-critical)'
                : engine.life_progress > 70
                  ? 'var(--color-warning)'
                  : 'var(--color-healthy)'
            }}
          />
        </div>
        <span className="progress-text">{engine.life_progress}%</span>
      </div>
    </div>
  )
}

// ─── Engine Detail Panel ──────────────────────────────────────────────

function EngineDetailPanel({ engine, history }) {
  if (!engine) {
    return (
      <div className="detail-panel empty">
        <div className="detail-empty">
          <span className="detail-empty-icon">✈️</span>
          <p>Select an engine to view detailed telemetry</p>
        </div>
      </div>
    )
  }

  const sensors = engine.sensors || {}
  const sensorEntries = Object.entries(sensors)

  // Build sparkline data from history
  const getSparkData = (sensorName) => {
    if (!history || !history.length) return []
    return history.slice(-30).map(r => r.sensors?.[sensorName] || 0)
  }

  return (
    <div className="detail-panel" id="engine-detail-panel">
      {/* Header */}
      <div className="detail-header">
        <div>
          <h2>{engine.name || `Engine #${engine.engine_id}`}</h2>
          <span className="detail-subtitle">Cycle {engine.current_cycle} / {engine.max_life}</span>
        </div>
        <span
          className="detail-health-badge"
          style={{ color: getHealthColor(engine.health_status) }}
        >
          {getHealthIcon(engine.health_status)} {engine.health_status.toUpperCase()}
        </span>
      </div>

      {/* Key Metrics Row */}
      <div className="detail-gauges">
        <RadialGauge
          value={engine.rul_prediction}
          max={200}
          label="RUL"
          unit="cycles"
          color={engine.rul_prediction < 30 ? 'var(--color-critical)' : engine.rul_prediction < 60 ? 'var(--color-warning)' : 'var(--color-healthy)'}
          size={110}
        />
        <RadialGauge
          value={engine.anomaly_score * 100}
          max={100}
          label="Anomaly"
          unit="%"
          color={engine.anomaly_score > 0.65 ? 'var(--color-critical)' : engine.anomaly_score > 0.4 ? 'var(--color-warning)' : 'var(--color-healthy)'}
          size={110}
        />
        <RadialGauge
          value={engine.life_progress}
          max={100}
          label="Life Used"
          unit="%"
          color={engine.life_progress > 85 ? 'var(--color-critical)' : engine.life_progress > 70 ? 'var(--color-warning)' : 'var(--color-accent)'}
          size={110}
        />
      </div>

      {/* Sensor Grid */}
      <div className="sensor-grid">
        <h3 className="section-title">Sensor Telemetry</h3>
        <div className="sensors">
          {sensorEntries.map(([name, value]) => (
            <div key={name} className="sensor-item">
              <div className="sensor-header">
                <span className="sensor-name">{name}</span>
                <span className="sensor-value">{typeof value === 'number' ? value.toFixed(2) : value}</span>
              </div>
              <Sparkline
                data={getSparkData(name)}
                color="var(--color-accent)"
                width={140}
                height={28}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Anomaly Alert */}
      {engine.is_anomaly && (
        <div className="anomaly-alert">
          <div className="alert-icon">⚠️</div>
          <div className="alert-content">
            <strong>{engine.rul_prediction < 30 ? 'CRITICAL: RUL Below 30 Cycles' : 'Anomaly Detected'}</strong>
            <p>Anomaly score: {(engine.anomaly_score * 100).toFixed(1)}% — {engine.trend === 'critical_degradation' ? 'Critical degradation detected. Immediate inspection recommended.' : 'Degradation trend detected. Schedule maintenance.'}</p>
            {engine.top_drifting_sensors && engine.top_drifting_sensors.length > 0 && (
              <div className="root-cause">
                <strong>Root Cause — Top Drifting Sensors:</strong>
                <div className="drift-bars">
                  {engine.top_drifting_sensors.map((s, i) => (
                    <div key={s.sensor} className="drift-bar-row">
                      <span className="drift-rank">#{i + 1}</span>
                      <span className="drift-sensor">{s.sensor}</span>
                      <div className="drift-bar-bg">
                        <div
                          className="drift-bar-fill"
                          style={{ width: `${Math.min(s.drift_rate * 20, 100)}%` }}
                        />
                      </div>
                      <span className="drift-value">{s.drift_rate.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Fleet Overview Bar ───────────────────────────────────────────────

function FleetOverviewBar({ engines }) {
  const healthy = engines.filter(e => e.health_status === 'healthy').length
  const warning = engines.filter(e => e.health_status === 'warning').length
  const critical = engines.filter(e => e.health_status === 'critical').length

  return (
    <div className="fleet-overview" id="fleet-overview">
      <div className="fleet-stat">
        <span className="fleet-stat-value">{engines.length}</span>
        <span className="fleet-stat-label">Total Engines</span>
      </div>
      <div className="fleet-stat healthy">
        <span className="fleet-stat-value">{healthy}</span>
        <span className="fleet-stat-label">Healthy</span>
      </div>
      <div className="fleet-stat warning">
        <span className="fleet-stat-value">{warning}</span>
        <span className="fleet-stat-label">Warning</span>
      </div>
      <div className="fleet-stat critical">
        <span className="fleet-stat-value">{critical}</span>
        <span className="fleet-stat-label">Critical</span>
      </div>
      <div className="fleet-stat">
        <span className="fleet-stat-value">
          {engines.length > 0 ? Math.round(engines.reduce((s, e) => s + e.rul_prediction, 0) / engines.length) : 0}
        </span>
        <span className="fleet-stat-label">Avg RUL</span>
      </div>
    </div>
  )
}

// ─── Bandwidth Display ──────────────────────────────────────────────────

function BandwidthWidget({ metrics }) {
  if (!metrics) return null;
  return (
    <div className="latency-widget" id="latency-widget">
      <h3 className="section-title">SATCOM Bandwidth Optimizer</h3>
      <div className="latency-bars">
        <div className="latency-bar-row">
          <span className="latency-label">Raw Telemetry</span>
          <div className="latency-bar-container">
            <div className="latency-bar cloud" style={{ width: '100%' }} />
          </div>
          <span className="latency-value">{metrics.full_bytes} B/s</span>
        </div>
        <div className="latency-bar-row">
          <span className="latency-label">Edge Optimized</span>
          <div className="latency-bar-container">
            <div className="latency-bar quantized" style={{ width: `${Math.max(5, 100 - metrics.saved_percent)}%` }} />
          </div>
          <span className="latency-value">{metrics.optimized_bytes} B/s</span>
        </div>
      </div>
      <div style={{ marginTop: '1rem', textAlign: 'center' }}>
        <span style={{ fontSize: '1.5rem', color: 'var(--color-healthy)', fontWeight: 'bold' }}>
          {metrics.saved_percent}% Saved
        </span>
      </div>
      <p className="latency-note">
        ✈️ Edge AI analyzes high-freq data locally and only transmits anomalies via expensive SATCOM.
      </p>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────

function App() {
  const [engines, setEngines] = useState([])
  const [selectedEngineId, setSelectedEngineId] = useState(null)
  const [history, setHistory] = useState({})
  const [connected, setConnected] = useState(false)
  const [bandwidth, setBandwidth] = useState({ full_bytes: 5000, optimized_bytes: 50, saved_percent: 99.0 })
  const [lastUpdate, setLastUpdate] = useState(Date.now())
  const [timeSinceUpdate, setTimeSinceUpdate] = useState(0)
  const wsRef = useRef(null)

  // Heartbeat monitor
  useEffect(() => {
    const interval = setInterval(() => {
      setTimeSinceUpdate(Math.floor((Date.now() - lastUpdate) / 1000))
    }, 1000)
    return () => clearInterval(interval)
  }, [lastUpdate])

  // Server-Sent Events connection
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === EventSource.OPEN) return

    const sse = new EventSource(SSE_URL)

    sse.onopen = () => {
      setConnected(true)
      console.log('🛩️ AeroGuard SSE connected')
    }

    sse.addEventListener('fleet_update', (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.data) {
          setEngines(msg.data)
          setLastUpdate(Date.now())
          if (msg.bandwidth_metrics) {
            setBandwidth(msg.bandwidth_metrics)
          }
          // Accumulate history
          setHistory(prev => {
            const next = { ...prev }
            msg.data.forEach(engine => {
              const eid = engine.engine_id
              if (!next[eid]) next[eid] = []
              next[eid] = [...next[eid].slice(-99), {
                cycle: engine.current_cycle,
                sensors: engine.sensors,
                rul: engine.rul_prediction,
                anomaly: engine.anomaly_score,
              }]
            })
            return next
          })
        }
      } catch (e) {
        console.error('SSE parse error:', e)
      }
    })

    sse.onerror = () => {
      setConnected(false)
      sse.close()
      // Reconnect handled natively by EventSource, but we can reset state
      setTimeout(connectWs, 2000)
    }

    wsRef.current = sse
  }, [])

  useEffect(() => {
    connectWs()
    return () => {
      wsRef.current?.close()
    }
  }, [connectWs])

  // If no backend, use simulated data
  useEffect(() => {
    if (connected) return

    const simEngines = [
      { engine_id: 1, name: 'CFM56-7B #1', current_cycle: 85, max_life: 320, life_progress: 26.6, rul_prediction: 235, anomaly_score: 0.08, is_anomaly: false, health_status: 'healthy', trend: 'stable', model_used: 'simulated', top_drifting_sensors: [], sensors: { T2: 518.67, T24: 642.9, T30: 1590.2, T50: 1401.5, P30: 554.8, Nf: 2388.1, Nc: 9046.5, Ps30: 47.48, phi: 521.9, NRf: 2388.1, NRc: 8138.9, BPR: 8.449, P2: 14.62, P15: 21.61 } },
      { engine_id: 2, name: 'CFM56-7B #2', current_cycle: 195, max_life: 280, life_progress: 69.6, rul_prediction: 85, anomaly_score: 0.28, is_anomaly: false, health_status: 'healthy', trend: 'stable', model_used: 'simulated', top_drifting_sensors: [], sensors: { T2: 518.67, T24: 644.1, T30: 1593.8, T50: 1412.3, P30: 556.2, Nf: 2388.4, Nc: 9050.1, Ps30: 47.54, phi: 522.6, NRf: 2388.4, NRc: 8142.1, BPR: 8.451, P2: 14.62, P15: 21.61 } },
      { engine_id: 3, name: 'CFM56-7B #3', current_cycle: 218, max_life: 250, life_progress: 87.2, rul_prediction: 32, anomaly_score: 0.62, is_anomaly: false, health_status: 'warning', trend: 'degrading', model_used: 'simulated', top_drifting_sensors: [{sensor: 'T50', drift_rate: 3.2}, {sensor: 'T30', drift_rate: 1.8}, {sensor: 'Nc', drift_rate: 1.1}], sensors: { T2: 518.67, T24: 645.8, T30: 1598.4, T50: 1428.7, P30: 558.9, Nf: 2389.0, Nc: 9055.2, Ps30: 47.63, phi: 523.7, NRf: 2389.0, NRc: 8147.6, BPR: 8.454, P2: 14.62, P15: 21.61 } },
      { engine_id: 4, name: 'CFM56-7B #4', current_cycle: 210, max_life: 220, life_progress: 95.5, rul_prediction: 10, anomaly_score: 0.89, is_anomaly: true, health_status: 'critical', trend: 'critical_degradation', model_used: 'simulated', top_drifting_sensors: [{sensor: 'T50', drift_rate: 4.8}, {sensor: 'T30', drift_rate: 3.5}, {sensor: 'Nc', drift_rate: 2.9}], sensors: { T2: 518.67, T24: 648.2, T30: 1605.1, T50: 1452.9, P30: 562.3, Nf: 2389.8, Nc: 9062.4, Ps30: 47.78, phi: 525.4, NRf: 2389.8, NRc: 8155.1, BPR: 8.458, P2: 14.62, P15: 21.61 } },
    ]

    setEngines(simEngines)

    // Simulate streaming
    const interval = setInterval(() => {
      setEngines(prev => prev.map(e => {
        const newCycle = e.current_cycle + 1
        const progress = newCycle / e.max_life
        const newAnomaly = Math.min(1, Math.max(0, e.anomaly_score + (Math.random() - 0.48) * 0.02))
        const newRul = Math.max(0, e.rul_prediction - 1 + (Math.random() - 0.5) * 2)

        // Evolve sensors slightly
        const newSensors = {}
        Object.entries(e.sensors).forEach(([k, v]) => {
          newSensors[k] = v + (Math.random() - 0.48) * 0.3
        })

        return {
          ...e,
          current_cycle: newCycle >= e.max_life ? 30 : newCycle,
          life_progress: parseFloat(((newCycle >= e.max_life ? 30 : newCycle) / e.max_life * 100).toFixed(1)),
          rul_prediction: newRul,
          anomaly_score: newAnomaly,
          is_anomaly: newAnomaly > 0.65,
          health_status: progress > 0.85 ? 'critical' : progress > 0.75 ? 'warning' : 'healthy',
          sensors: newSensors,
        }
      }))
    }, 800)

    return () => clearInterval(interval)
  }, [connected])

  const selectedEngine = engines.find(e => e.engine_id === selectedEngineId)
  const engineHistory = history[selectedEngineId] || []

  return (
    <div className="app" id="aeroguard-app">
      {/* Header */}
      <header className="app-header" id="app-header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-icon">✈️</span>
            <h1>AeroGuard</h1>
          </div>
          <span className="tagline">Edge-Native Engine Health Intelligence</span>
        </div>
        <div className="header-right">
          <div className={`connection-status ${connected ? (timeSinceUpdate > 3 ? 'warning' : 'connected') : 'disconnected'}`}>
            <span className="status-dot" />
            {connected ? (timeSinceUpdate > 3 ? `HEARTBEAT LAGGING (${timeSinceUpdate}s)` : 'LIVE') : 'DEMO MODE'}
          </div>
          <span className="timestamp">{new Date().toLocaleTimeString()}</span>
        </div>
      </header>

      {/* Fleet Overview */}
      <FleetOverviewBar engines={engines} />

      {/* Main Content */}
      <main className="main-content" id="main-content">
        {/* Engine List */}
        <section className="engine-list" id="engine-list">
          <h2 className="section-heading">Fleet Engines</h2>
          {engines.map(engine => (
            <EngineCard
              key={engine.engine_id}
              engine={engine}
              isSelected={selectedEngineId === engine.engine_id}
              onClick={() => setSelectedEngineId(engine.engine_id)}
            />
          ))}

          {/* Latency Widget */}
          <BandwidthWidget metrics={bandwidth} />
        </section>

        {/* Detail Panel */}
        <section className="detail-section" id="detail-section">
          <EngineDetailPanel engine={selectedEngine} history={engineHistory} />
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <span>AeroGuard v2.0 — SATCOM Bandwidth Optimizer</span>
        <span>NASA C-MAPSS Dataset • XGBoost Edge AI • Redis State Management • SSE Streaming</span>
      </footer>
    </div>
  )
}

function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  )
}

export default AppWithErrorBoundary
