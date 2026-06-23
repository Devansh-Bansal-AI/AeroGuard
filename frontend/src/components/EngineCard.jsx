import { getHealthColor, getHealthIcon, formatRUL } from '../utils'

export default function EngineCard({ engine, isSelected, onClick }) {
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
              {engine.model_used.replace('_', ' ').toUpperCase()}
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
