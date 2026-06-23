import { useState, useEffect, useRef, useCallback, Component } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import DiagnosticCockpit from './pages/DiagnosticCockpit'
import EdgeMetrics from './pages/EdgeMetrics'
import { SSE_URL, API_URL, getHealthColor, getHealthIcon } from './utils'
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

import EngineCard from './components/EngineCard'
import EngineDetailPanel from './components/EngineDetailPanel'
import FleetOverviewBar from './components/FleetOverviewBar'
import BandwidthWidget from './components/BandwidthWidget'

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

    const interval = setInterval(() => {
      setEngines(prev => prev.map(e => {
        const newCycle = e.current_cycle + 1
        const progress = newCycle / e.max_life
        const newAnomaly = Math.min(1, Math.max(0, e.anomaly_score + (Math.random() - 0.48) * 0.02))
        const newRul = Math.max(0, e.rul_prediction - 1 + (Math.random() - 0.5) * 2)

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
          <nav className="nav-tabs">
            <NavLink to="/" end className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
              Fleet Dashboard
            </NavLink>
            <NavLink to="/diagnostics" className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
              Diagnostics
            </NavLink>
            <NavLink to="/edge" className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
              Edge Metrics
            </NavLink>
          </nav>
        </div>
        <div className="header-right">
          <div className={`connection-status ${connected ? (timeSinceUpdate > 3 ? 'warning' : 'connected') : 'disconnected'}`}>
            <span className="status-dot" />
            {connected ? (timeSinceUpdate > 3 ? `HEARTBEAT LAGGING (${timeSinceUpdate}s)` : 'LIVE') : 'DEMO MODE'}
          </div>
          <span className="timestamp">{new Date().toLocaleTimeString()}</span>
        </div>
      </header>

      <Routes>
        <Route path="/" element={
          <>
            <FleetOverviewBar engines={engines} />
            <main className="main-content" id="main-content">
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
                <BandwidthWidget metrics={bandwidth} />
              </section>
              <section className="detail-section" id="detail-section">
                <EngineDetailPanel engine={selectedEngine} history={engineHistory} />
              </section>
            </main>
          </>
        } />
        <Route path="/diagnostics" element={<DiagnosticCockpit />} />
        <Route path="/edge" element={<EdgeMetrics />} />
      </Routes>

      {/* Footer */}
      <footer className="app-footer">
        <span>AeroGuard v2.0 — Edge-Native Engine Health Intelligence</span>
        <span>NASA C-MAPSS Dataset • BiLSTM+Attention • ONNX INT8 Edge AI • SSE Streaming</span>
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

