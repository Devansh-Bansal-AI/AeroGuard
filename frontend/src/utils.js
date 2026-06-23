// Shared utility functions for AeroGuard dashboard
export const SSE_URL = import.meta.env.VITE_SSE_URL || 'http://localhost:8000/ws/stream'
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export function getHealthColor(status) {
  switch (status) {
    case 'healthy': return 'var(--color-healthy)'
    case 'warning': return 'var(--color-warning)'
    case 'critical': return 'var(--color-critical)'
    default: return 'var(--color-text-dim)'
  }
}

export function getHealthIcon(status) {
  switch (status) {
    case 'healthy': return '●'
    case 'warning': return '◆'
    case 'critical': return '▲'
    default: return '○'
  }
}

export function formatRUL(rul) {
  return Math.round(rul)
}
