import { API_BASE_URL, ENDPOINTS } from "./config";

// ─── Helpers ─────────────────────────────────────────────────────────────────
const getAuthHeaders = () => {
  const token = localStorage.getItem("token");
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

const handleResponse = async (res) => {
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const err = await res.json();
      msg = err.message || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
};

// ─── Auth API ─────────────────────────────────────────────────────────────────
export const authApi = {
  /**
   * POST /auth/register
   * Body: { first_name, last_name, email, password }
   */
  register: async ({ firstName, lastName, email, password }) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.REGISTER}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        first_name: firstName,
        last_name: lastName,
        email,
        password,
      }),
    });
    return handleResponse(res);
  },

  /**
   * POST /auth/login
   * Body: { first_name, last_name, email, password }
   * Expected response: { token, user: { id, firstName, lastName, email } }
   */
  login: async ({ firstName, lastName, email, password }) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.LOGIN}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        first_name: firstName,
        last_name: lastName,
        email,
        password,
      }),
    });
    return handleResponse(res);
  },

  logout: () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  },
};

// ─── Camera API ───────────────────────────────────────────────────────────────
export const cameraApi = {
  /**
   * GET /cameras
   * Returns: [{ id, name, stream_path }]
   */
  getCameras: async () => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.CAMERAS}`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  /**
   * POST /cameras/add
   * Body: { name, stream_path, count_axis, in_direction, shot_type, processing_preset, counting_enabled, density_enabled, monitored_area_sqm }
   * Returns: created camera object with processing/counting/density settings
   */
  addCamera: async ({
    name,
    streamPath,
    countAxis,
    inDirection,
    shotType,
    processingPreset,
    countingEnabled,
    densityEnabled,
    monitoredAreaSqm,
  }) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.ADD_CAMERA}`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        name,
        stream_path: streamPath,
        count_axis: countAxis || "x",
        in_direction: inDirection || "positive",
        shot_type: shotType || "ground",
        processing_preset: processingPreset || "balanced",
        counting_enabled: countingEnabled ?? true,
        density_enabled: densityEnabled ?? false,
        monitored_area_sqm: densityEnabled ? monitoredAreaSqm : null,
      }),
    });
    return handleResponse(res);
  },

  /**
   * GET /cameras/counts
   * Returns: [{ camera_id, count, timestamp }]
   * Your backend updates these every 10 minutes
   */
  getAllCameraCounts: async () => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.CAMERA_COUNTS}`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  /**
   * GET /cameras/counts/:cameraId
   * Returns: { camera_id, count, timestamp }
   */
  getCameraCount: async (cameraId) => {
    const res = await fetch(
      `${API_BASE_URL}${ENDPOINTS.CAMERA_COUNTS}/${cameraId}`,
      {
        headers: getAuthHeaders(),
      },
    );
    return handleResponse(res);
  },

  /**
   * Builds MJPEG URL for camera live stream.
   * GET /cameras/stream/:cameraId?overlay=true|false
   */
  getCameraStreamUrl: (cameraId, { overlay = true } = {}) => {
    const encodedId = encodeURIComponent(cameraId);
    return `${API_BASE_URL}${ENDPOINTS.CAMERA_STREAM}/${encodedId}?overlay=${overlay ? "true" : "false"}`;
  },

  /**
   * DELETE /cameras/:id
   */
  deleteCamera: async (cameraId) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.CAMERAS}/${cameraId}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  /**
   * POST /cameras/reset/:cameraId
   * Resets the count for a specific camera to 0
   */
  resetCameraCount: async (cameraId) => {
    const res = await fetch(
      `${API_BASE_URL}${ENDPOINTS.RESET_COUNT}/${cameraId}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
      },
    );
    return handleResponse(res);
  },

  /**
   * POST /cameras/reset-all
   * Resets the count for all cameras to 0
   */
  resetAllCameraCounts: async () => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.RESET_ALL_COUNTS}`, {
      method: "POST",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  /**
   * POST /cameras/start/:cameraId
   * Starts monitoring for a specific camera
   */
  startCamera: async (cameraId) => {
    const res = await fetch(
      `${API_BASE_URL}${ENDPOINTS.CAMERA_START}/${cameraId}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
      },
    );
    return handleResponse(res);
  },

  /**
   * POST /cameras/stop/:cameraId
   * Stops monitoring for a specific camera
   */
  stopCamera: async (cameraId) => {
    const res = await fetch(
      `${API_BASE_URL}${ENDPOINTS.CAMERA_STOP}/${cameraId}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
      },
    );
    return handleResponse(res);
  },
};
