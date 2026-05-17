import { CAMERA_API_URLS } from "./config";

export const getCameraBackendUrl = (camera) => camera?.backendUrl || camera?.backend_url;

export const buildCameraBackendQueue = (cameras = []) => {
  const activeBackends = new Set(cameras.map(getCameraBackendUrl).filter(Boolean));
  return CAMERA_API_URLS.filter((backendUrl) => !activeBackends.has(backendUrl));
};

export const reserveCameraBackend = (queue, backendUrl) =>
  queue.filter((queuedUrl) => queuedUrl !== backendUrl);

export const releaseCameraBackend = (queue, backendUrl, cameras = []) => {
  if (!backendUrl || !CAMERA_API_URLS.includes(backendUrl)) {
    return queue;
  }

  const activeBackends = new Set(cameras.map(getCameraBackendUrl).filter(Boolean));
  if (activeBackends.has(backendUrl) || queue.includes(backendUrl)) {
    return queue;
  }

  const configuredIndex = CAMERA_API_URLS.indexOf(backendUrl);
  const insertAt = queue.findIndex(
    (queuedUrl) => CAMERA_API_URLS.indexOf(queuedUrl) > configuredIndex,
  );

  if (insertAt === -1) {
    return [...queue, backendUrl];
  }
  return [...queue.slice(0, insertAt), backendUrl, ...queue.slice(insertAt)];
};
