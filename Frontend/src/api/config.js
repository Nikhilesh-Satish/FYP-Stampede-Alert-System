// ─── API Base URL ────────────────────────────────────────────────────────────
// Set VITE_API_URL in your .env file. Defaults to localhost:8000
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ─── Endpoint Paths ──────────────────────────────────────────────────────────
export const ENDPOINTS = {
  LOGIN:          '/auth/login',
  REGISTER:       '/auth/register',
  CAMERAS:        '/cameras',
  ADD_CAMERA:     '/cameras/add',
  CAMERA_COUNTS:  '/cameras/counts',   // GET /cameras/counts  or  /cameras/counts/:id
}

// ─── Alert Thresholds ────────────────────────────────────────────────────────
// Modify these values to change when alerts trigger
export const THRESHOLDS = {
  SAFE:     { max: 100,      label: 'Safe',     color: '#22c55e', bg: 'rgba(34,197,94,0.12)' },
  WARNING:  { max: 200,      label: 'Warning',  color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' },
  DANGER:   { max: 300,      label: 'Danger',   color: '#ef4444', bg: 'rgba(239,68,68,0.12)' },
  CRITICAL: { max: Infinity, label: 'Critical', color: '#dc2626', bg: 'rgba(220,38,38,0.2)'  },
}

export const getAlertLevel = (count) => {
  if (count <= THRESHOLDS.SAFE.max)    return { ...THRESHOLDS.SAFE,     level: 'SAFE'     }
  if (count <= THRESHOLDS.WARNING.max) return { ...THRESHOLDS.WARNING,  level: 'WARNING'  }
  if (count <= THRESHOLDS.DANGER.max)  return { ...THRESHOLDS.DANGER,   level: 'DANGER'   }
  return                                      { ...THRESHOLDS.CRITICAL, level: 'CRITICAL' }
}
