import { API_BASE_URL, CAMERA_API_URLS, ENDPOINTS } from "./config";

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

const fetchWithTimeout = async (url, options = {}, timeoutMs = 120000) => {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
};

const getGoogleDriveFileId = (url) => {
  const value = String(url || "").trim();
  const patterns = [
    /drive\.google\.com\/file\/d\/([^/]+)/,
    /drive\.google\.com\/open\?id=([^&]+)/,
    /drive\.google\.com\/uc\?(?:.*&)?id=([^&]+)/,
  ];

  for (const pattern of patterns) {
    const match = value.match(pattern);
    if (match?.[1]) return decodeURIComponent(match[1]);
  }

  return "";
};

const normalizeStreamPath = (streamPath) => {
  const trimmed = String(streamPath || "").trim();
  const driveFileId = getGoogleDriveFileId(trimmed);
  if (!driveFileId) return trimmed;
  return `https://drive.google.com/uc?export=download&id=${encodeURIComponent(driveFileId)}`;
};

const withBackend = (item, backendUrl) => ({
  ...item,
  backendUrl,
  backend_url: backendUrl,
  camera_key: `${backendUrl}::${item.camera_id || item.id || item.name}`,
});

const getBackendUrl = (cameraOrUrl) => {
  if (typeof cameraOrUrl === "string" && cameraOrUrl.startsWith("http")) {
    return cameraOrUrl.replace(/\/+$/, "");
  }
  return (
    cameraOrUrl?.backendUrl ||
    cameraOrUrl?.backend_url ||
    CAMERA_API_URLS[0] ||
    API_BASE_URL
  );
};

const getCameraId = (cameraOrId) =>
  typeof cameraOrId === "object" ? cameraOrId.id || cameraOrId.camera_id : cameraOrId;

const uniqueUrls = (urls) => [...new Set(urls.filter(Boolean).map(getBackendUrl))];

export const getCameraKey = (cameraOrCount) =>
  cameraOrCount?.camera_key ||
  `${getBackendUrl(cameraOrCount)}::${cameraOrCount?.camera_id || cameraOrCount?.id}`;

export { CAMERA_API_URLS };

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
    const results = await Promise.allSettled(
      CAMERA_API_URLS.map(async (backendUrl) => {
        const res = await fetch(`${backendUrl}${ENDPOINTS.CAMERAS}`, {
          headers: getAuthHeaders(),
        });
        const data = await handleResponse(res);
        const cameras = Array.isArray(data) ? data : data.cameras || [];
        return cameras.map((camera) => withBackend(camera, backendUrl));
      }),
    );

    const cameras = [];
    const failures = [];
    results.forEach((result, index) => {
      if (result.status === "fulfilled") {
        cameras.push(...result.value);
      } else {
        failures.push(`${CAMERA_API_URLS[index]}: ${result.reason.message}`);
      }
    });

    if (cameras.length === 0 && failures.length > 0) {
      throw new Error(failures.join("; "));
    }

    return cameras;
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
    backendUrl,
    backendUrls,
  }) => {
    const targetBackends = uniqueUrls([
      backendUrl,
      ...(Array.isArray(backendUrls) ? backendUrls : []),
    ]);
    if (targetBackends.length === 0) {
      throw new Error("No processing backend is available.");
    }
    const body = JSON.stringify({
        name,
        stream_path: normalizeStreamPath(streamPath),
        count_axis: countAxis || "x",
        in_direction: inDirection || "positive",
        shot_type: shotType || "ground",
        processing_preset: processingPreset || "balanced",
        counting_enabled: countingEnabled ?? true,
        density_enabled: densityEnabled ?? false,
        monitored_area_sqm: densityEnabled ? monitoredAreaSqm : null,
    });
    const failures = [];

    for (const targetBackend of targetBackends) {
      try {
        const res = await fetchWithTimeout(`${targetBackend}${ENDPOINTS.ADD_CAMERA}`, {
          method: "POST",
          headers: getAuthHeaders(),
          body,
        }, 45000);
        const data = await handleResponse(res);
        return withBackend(data, targetBackend);
      } catch (err) {
        const message =
          err.name === "AbortError"
            ? "timed out while starting backend"
            : err.message;
        failures.push(`${targetBackend}: ${message}`);
      }
    }

    throw new Error(failures.join("; ") || "Failed to add camera.");
  },

  /**
   * GET /cameras/counts
   * Returns: [{ camera_id, count, timestamp }]
   * Your backend updates these every 10 minutes
   */
  getAllCameraCounts: async (cameras = []) => {
    const backendUrls = cameras.length
      ? [...new Set(cameras.map(getBackendUrl))]
      : CAMERA_API_URLS;

    const results = await Promise.allSettled(
      backendUrls.map(async (backendUrl) => {
        const res = await fetch(`${backendUrl}${ENDPOINTS.CAMERA_COUNTS}`, {
          headers: getAuthHeaders(),
        });
        const data = await handleResponse(res);
        return (Array.isArray(data) ? data : []).map((count) =>
          withBackend(count, backendUrl),
        );
      }),
    );

    const counts = [];
    const failures = [];
    results.forEach((result, index) => {
      if (result.status === "fulfilled") {
        counts.push(...result.value);
      } else {
        failures.push(`${backendUrls[index]}: ${result.reason.message}`);
      }
    });

    if (counts.length === 0 && failures.length > 0) {
      throw new Error(failures.join("; "));
    }

    return counts;
  },

  /**
   * GET /cameras/counts/:cameraId
   * Returns: { camera_id, count, timestamp }
   */
  getCameraCount: async (cameraId) => {
    const targetBackend = getBackendUrl(cameraId);
    const id = encodeURIComponent(getCameraId(cameraId));
    const res = await fetch(
      `${targetBackend}${ENDPOINTS.CAMERA_COUNTS}/${id}`,
      {
        headers: getAuthHeaders(),
      },
    );
    const data = await handleResponse(res);
    return withBackend(data, targetBackend);
  },

  /**
   * Builds MJPEG URL for camera live stream.
   * GET /cameras/stream/:cameraId?overlay=true|false
   */
  getCameraStreamUrl: (cameraId, { overlay = true } = {}) => {
    const targetBackend = getBackendUrl(cameraId);
    const encodedId = encodeURIComponent(getCameraId(cameraId));
    return `${targetBackend}${ENDPOINTS.CAMERA_STREAM}/${encodedId}?overlay=${overlay ? "true" : "false"}`;
  },

  /**
   * DELETE /cameras/:id
   */
  deleteCamera: async (cameraId) => {
    const targetBackend = getBackendUrl(cameraId);
    const id = encodeURIComponent(getCameraId(cameraId));
    const res = await fetch(`${targetBackend}${ENDPOINTS.CAMERAS}/${id}`, {
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
    const targetBackend = getBackendUrl(cameraId);
    const id = encodeURIComponent(getCameraId(cameraId));
    const res = await fetch(
      `${targetBackend}${ENDPOINTS.RESET_COUNT}/${id}`,
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
    const results = await Promise.allSettled(
      CAMERA_API_URLS.map(async (backendUrl) => {
        const res = await fetch(`${backendUrl}${ENDPOINTS.RESET_ALL_COUNTS}`, {
          method: "POST",
          headers: getAuthHeaders(),
        });
        return handleResponse(res);
      }),
    );

    const failed = results.find((result) => result.status === "rejected");
    if (failed) throw failed.reason;
    return { message: "All camera backends reset" };
  },

  /**
   * POST /cameras/start/:cameraId
   * Starts monitoring for a specific camera
   */
  startCamera: async (cameraId) => {
    const targetBackend = getBackendUrl(cameraId);
    const id = encodeURIComponent(getCameraId(cameraId));
    const res = await fetch(
      `${targetBackend}${ENDPOINTS.CAMERA_START}/${id}`,
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
    const targetBackend = getBackendUrl(cameraId);
    const id = encodeURIComponent(getCameraId(cameraId));
    const res = await fetch(
      `${targetBackend}${ENDPOINTS.CAMERA_STOP}/${id}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
      },
    );
    return handleResponse(res);
  },
};
