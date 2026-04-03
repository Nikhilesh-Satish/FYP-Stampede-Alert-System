// ─── API Base URL ────────────────────────────────────────────────────────────
// Set VITE_API_URL in your .env file. Defaults to localhost:8000
export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── Endpoint Paths ──────────────────────────────────────────────────────────
export const ENDPOINTS = {
  LOGIN: "/auth/login",
  REGISTER: "/auth/register",
  CAMERAS: "/cameras",
  ADD_CAMERA: "/cameras/add",
  CAMERA_COUNTS: "/cameras/counts", // GET /cameras/counts  or  /cameras/counts/:id
  CAMERA_STREAM: "/cameras/stream", // GET /cameras/stream/:id?overlay=true|false (MJPEG)
  RESET_COUNT: "/cameras/reset", // POST /cameras/reset/:id
  RESET_ALL_COUNTS: "/cameras/reset-all", // POST /cameras/reset-all
  CAMERA_START: "/cameras/start", // POST /cameras/start/:id
  CAMERA_STOP: "/cameras/stop", // POST /cameras/stop/:id
};

// ─── GLOBAL CAPACITY SETTING ──────────────────────────────────────────────────
// Default capacity. This can be changed in the UI dashboard.
export const DEFAULT_AREA_CAPACITY = 500;

// ─── Alert Thresholds ────────────────────────────────────────────────────────
// Percentage-based thresholds relative to room capacity
export const CAPACITY_THRESHOLDS = {
  SAFE: {
    maxPercent: 60,
    label: "Safe",
    color: "#22c55e",
    bg: "rgba(34,197,94,0.12)",
  },
  WARNING: {
    maxPercent: 80,
    label: "Warning",
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.12)",
  },
  DANGER: {
    maxPercent: 95,
    label: "Danger",
    color: "#ef4444",
    bg: "rgba(239,68,68,0.12)",
  },
  CRITICAL: {
    maxPercent: 100,
    label: "Critical",
    color: "#dc2626",
    bg: "rgba(220,38,38,0.2)",
  },
};

export const getAlertLevel = (count, capacity) => {
  // Use default capacity if not provided
  const finalCapacity = capacity || DEFAULT_AREA_CAPACITY;
  const percent = (count / finalCapacity) * 100;

  if (percent <= CAPACITY_THRESHOLDS.SAFE.maxPercent)
    return { ...CAPACITY_THRESHOLDS.SAFE, level: "SAFE" };
  if (percent <= CAPACITY_THRESHOLDS.WARNING.maxPercent)
    return { ...CAPACITY_THRESHOLDS.WARNING, level: "WARNING" };
  if (percent <= CAPACITY_THRESHOLDS.DANGER.maxPercent)
    return { ...CAPACITY_THRESHOLDS.DANGER, level: "DANGER" };
  return { ...CAPACITY_THRESHOLDS.CRITICAL, level: "CRITICAL" };
};

export const DENSITY_THRESHOLDS = {
  SAFE: {
    maxScore: 2.0,
    label: "Safe",
    color: "#22c55e",
    bg: "rgba(34,197,94,0.12)",
  },
  WARNING: {
    maxScore: 3.5,
    label: "Warning",
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.12)",
  },
  DANGER: {
    maxScore: 5.0,
    label: "Danger",
    color: "#ef4444",
    bg: "rgba(239,68,68,0.12)",
  },
  CRITICAL: {
    maxScore: Number.POSITIVE_INFINITY,
    label: "Critical",
    color: "#dc2626",
    bg: "rgba(220,38,38,0.2)",
  },
};

export const getDensityAlertLevel = (densityScore) => {
  const score = Number.isFinite(Number(densityScore)) ? Number(densityScore) : 0;
  if (score <= DENSITY_THRESHOLDS.SAFE.maxScore)
    return { ...DENSITY_THRESHOLDS.SAFE, level: "SAFE" };
  if (score <= DENSITY_THRESHOLDS.WARNING.maxScore)
    return { ...DENSITY_THRESHOLDS.WARNING, level: "WARNING" };
  if (score <= DENSITY_THRESHOLDS.DANGER.maxScore)
    return { ...DENSITY_THRESHOLDS.DANGER, level: "DANGER" };
  return { ...DENSITY_THRESHOLDS.CRITICAL, level: "CRITICAL" };
};
