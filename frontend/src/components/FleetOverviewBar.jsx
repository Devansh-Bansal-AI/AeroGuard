export default function FleetOverviewBar({ engines }) {
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
