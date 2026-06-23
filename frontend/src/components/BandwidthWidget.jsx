export default function BandwidthWidget({ metrics }) {
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
