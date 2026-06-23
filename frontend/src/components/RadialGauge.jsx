export default function RadialGauge({ value, max, label, unit, color, size = 120 }) {
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
