import { useState, useEffect } from 'react'
import { API_URL, getHealthColor } from '../utils'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'

export default function DiagnosticCockpit() {
  const [engines, setEngines] = useState([])
  const [selectedId, setSelectedId] = useState(1)
  const [explanation, setExplanation] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  // Fetch fleet data
  useEffect(() => {
    const fetchFleet = async () => {
      try {
        const res = await fetch(`${API_URL}/fleet`)
        const data = await res.json()
        setEngines(data.engines || [])
        setLoading(false)
      } catch {
        setLoading(false)
      }
    }
    fetchFleet()
    const interval = setInterval(fetchFleet, 3000)
    return () => clearInterval(interval)
  }, [])

  // Fetch explainability data when engine changes
  useEffect(() => {
    const fetchExplain = async () => {
      try {
        const res = await fetch(`${API_URL}/engine/${selectedId}/explain`)
        const data = await res.json()
        if (data.available) {
          setExplanation(data)
        } else {
          setExplanation(null)
        }
      } catch {
        setExplanation(null)
      }
    }

    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_URL}/engine/${selectedId}/history?last_n=50`)
        const data = await res.json()
        if (data.readings && Array.isArray(data.readings)) {
          setHistory(data.readings)
        } else if (Array.isArray(data)) {
          setHistory(data)
        }
      } catch {
        setHistory([])
      }
    }

    fetchExplain()
    fetchHistory()
  }, [selectedId])

  const selectedEngine = engines.find(e => e.engine_id === selectedId) || {}

  // Build RUL trend data from history
  const rulTrend = history.map((r, i) => ({
    cycle: r.cycle || i,
    rul: r.rul_prediction || r.rul || 0,
    anomaly: (r.anomaly_score || r.anomaly || 0) * 100,
  }))

  // Build attention heatmap data
  const attentionData = explanation?.attention_weights?.map((w, i) => ({
    timestep: i + 1,
    weight: w,
  })) || []

  // Build sensor comparison radar
  const sensorKeys = Object.keys(selectedEngine.sensors || {}).slice(0, 8)
  const sensorRadar = sensorKeys.map(k => ({
    sensor: k,
    value: Math.min(100, Math.abs(selectedEngine.sensors[k] || 0) / 20),
  }))

  return (
    <div className="diagnostic-page">
      <div className="page-header">
        <h1>🔬 Diagnostic Cockpit</h1>
        <p className="page-subtitle">Deep engine analytics, attention explainability & sensor forensics</p>
      </div>

      {/* Engine Selector Tabs */}
      <div className="engine-tabs">
        {engines.map(e => (
          <button
            key={e.engine_id}
            className={`engine-tab ${selectedId === e.engine_id ? 'active' : ''} ${e.health_status}`}
            onClick={() => setSelectedId(e.engine_id)}
          >
            <span className="tab-indicator" style={{ color: getHealthColor(e.health_status) }}>●</span>
            <span>{e.name || `Engine #${e.engine_id}`}</span>
            <span className="tab-rul">{Math.round(e.rul_prediction)} cyc</span>
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading-state">
          <div className="loading-spinner" />
          <p>Connecting to AeroGuard backend...</p>
        </div>
      ) : (
        <div className="diagnostic-grid">
          {/* RUL Trend Chart */}
          <div className="chart-card wide">
            <h3 className="chart-title">📈 RUL & Anomaly Trend</h3>
            {rulTrend.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={rulTrend}>
                  <defs>
                    <linearGradient id="rulGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#00e5ff" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="anomGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff4d4d" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ff4d4d" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1f2e" />
                  <XAxis dataKey="cycle" stroke="#555" fontSize={11} />
                  <YAxis yAxisId="rul" stroke="#00e5ff" fontSize={11} />
                  <YAxis yAxisId="anomaly" orientation="right" stroke="#ff4d4d" fontSize={11} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ background: '#0d1117', border: '1px solid #1a1f2e', borderRadius: 8 }}
                    labelStyle={{ color: '#888' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Area yAxisId="rul" type="monotone" dataKey="rul" stroke="#00e5ff" fill="url(#rulGrad)" strokeWidth={2} name="RUL (cycles)" />
                  <Area yAxisId="anomaly" type="monotone" dataKey="anomaly" stroke="#ff4d4d" fill="url(#anomGrad)" strokeWidth={2} name="Anomaly (%)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">No trend data available. Connect to backend.</div>
            )}
          </div>

          {/* Attention Heatmap */}
          <div className="chart-card">
            <h3 className="chart-title">🧠 Attention Heatmap</h3>
            <p className="chart-desc">Which timesteps did the model focus on for RUL prediction?</p>
            {attentionData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={attentionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a1f2e" />
                    <XAxis dataKey="timestep" stroke="#555" fontSize={10} interval={4} />
                    <YAxis stroke="#555" fontSize={10} />
                    <Tooltip
                      contentStyle={{ background: '#0d1117', border: '1px solid #1a1f2e', borderRadius: 8 }}
                      formatter={(val) => [val.toFixed(4), 'Attention']}
                    />
                    <Bar dataKey="weight" fill="#7c3aed" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="attention-summary">
                  <div className="top-timesteps">
                    <strong>Top focus areas:</strong>
                    {explanation?.top_timesteps?.slice(0, 5).map((t, i) => (
                      <span key={i} className="timestep-badge">
                        t={t.timestep} ({(t.weight * 100).toFixed(1)}%)
                      </span>
                    ))}
                  </div>
                  <p className="attention-note">{explanation?.interpretation}</p>
                </div>
              </>
            ) : (
              <div className="chart-empty">
                <p>Attention data not available.</p>
                <p className="chart-empty-hint">Requires 50+ sensor readings and PyTorch model.</p>
              </div>
            )}
          </div>

          {/* Sensor Radar */}
          <div className="chart-card">
            <h3 className="chart-title">📡 Sensor Profile</h3>
            {sensorRadar.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={sensorRadar} outerRadius={90}>
                  <PolarGrid stroke="#1a1f2e" />
                  <PolarAngleAxis dataKey="sensor" stroke="#888" fontSize={11} />
                  <PolarRadiusAxis stroke="#333" fontSize={10} />
                  <Radar name="Sensors" dataKey="value" stroke="#00e5ff" fill="#00e5ff" fillOpacity={0.15} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">No sensor data</div>
            )}
          </div>

          {/* Engine Vitals Card */}
          <div className="chart-card vitals">
            <h3 className="chart-title">⚡ Engine Vitals</h3>
            <div className="vitals-grid">
              <div className="vital">
                <span className="vital-label">RUL</span>
                <span className="vital-value" style={{ color: selectedEngine.rul_prediction < 30 ? '#ff4d4d' : '#00e5ff' }}>
                  {Math.round(selectedEngine.rul_prediction || 0)}
                </span>
                <span className="vital-unit">cycles</span>
              </div>
              <div className="vital">
                <span className="vital-label">Anomaly</span>
                <span className="vital-value" style={{ color: selectedEngine.anomaly_score > 0.5 ? '#ff4d4d' : '#22c55e' }}>
                  {((selectedEngine.anomaly_score || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="vital">
                <span className="vital-label">Life Used</span>
                <span className="vital-value">{selectedEngine.life_progress || 0}%</span>
              </div>
              <div className="vital">
                <span className="vital-label">Status</span>
                <span className="vital-value" style={{ color: getHealthColor(selectedEngine.health_status), fontSize: '0.9rem' }}>
                  {(selectedEngine.health_status || 'unknown').toUpperCase()}
                </span>
              </div>
              <div className="vital">
                <span className="vital-label">Model</span>
                <span className="vital-value" style={{ fontSize: '0.8rem', color: '#7c3aed' }}>
                  {(selectedEngine.model_used || 'simulated').replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <div className="vital">
                <span className="vital-label">Cycle</span>
                <span className="vital-value">{selectedEngine.current_cycle || 0}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
