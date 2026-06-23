import { getHealthColor, getHealthIcon } from '../utils'
import RadialGauge from './RadialGauge'
import Sparkline from './Sparkline'

export default function EngineDetailPanel({ engine, history }) {
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
