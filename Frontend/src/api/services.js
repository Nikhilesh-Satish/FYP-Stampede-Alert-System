import { API_BASE_URL, ENDPOINTS } from './config'

// ─── Helpers ─────────────────────────────────────────────────────────────────
const getAuthHeaders = () => {
  const token = localStorage.getItem('token')
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

const handleResponse = async (res) => {
  if (!res.ok) {
    let msg = `Request failed (${res.status})`
    try { const err = await res.json(); msg = err.message || msg } catch {}
    throw new Error(msg)
  }
  return res.json()
}

// ─── Auth API ─────────────────────────────────────────────────────────────────
export const authApi = {
  /**
   * POST /auth/register
   * Body: { first_name, last_name, email, password }
   */
  register: async ({ firstName, lastName, email, password }) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.REGISTER}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ first_name: firstName, last_name: lastName, email, password }),
    })
    return handleResponse(res)
  },

  /**
   * POST /auth/login
   * Body: { first_name, last_name, email, password }
   * Expected response: { token, user: { id, firstName, lastName, email } }
   */
  login: async ({ firstName, lastName, email, password }) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.LOGIN}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ first_name: firstName, last_name: lastName, email, password }),
    })
    return handleResponse(res)
  },

  logout: () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
  },
}

// ─── Camera API ───────────────────────────────────────────────────────────────
export const cameraApi = {
  /**
   * GET /cameras
   * Returns: [{ id, name, stream_path }]
   */
  getCameras: async () => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.CAMERAS}`, {
      headers: getAuthHeaders(),
    })
    return handleResponse(res)
  },

  /**
   * POST /cameras/add
   * Body: { name, stream_path }
   * Returns: created camera object { id, name, stream_path }
   */
  addCamera: async ({ name, streamPath }) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.ADD_CAMERA}`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ name, stream_path: streamPath }),
    })
    return handleResponse(res)
  },

  /**
   * GET /cameras/counts
   * Returns: [{ camera_id, count, timestamp }]
   * Your backend updates these every 10 minutes
   */
  getAllCameraCounts: async () => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.CAMERA_COUNTS}`, {
      headers: getAuthHeaders(),
    })
    return handleResponse(res)
  },

  /**
   * GET /cameras/counts/:cameraId
   * Returns: { camera_id, count, timestamp }
   */
  getCameraCount: async (cameraId) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.CAMERA_COUNTS}/${cameraId}`, {
      headers: getAuthHeaders(),
    })
    return handleResponse(res)
  },

  /**
   * DELETE /cameras/:id
   */
  deleteCamera: async (cameraId) => {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.CAMERAS}/${cameraId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    })
    return handleResponse(res)
  },
}
