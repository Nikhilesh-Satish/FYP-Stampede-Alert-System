import { useState, useEffect, useCallback, useRef } from "react";
import { cameraApi } from "../api/services";
import { getAlertLevel, DEFAULT_AREA_CAPACITY } from "../api/config";

// Polls camera counts from your API every 10 seconds (synced with camera worker)
const POLL_INTERVAL_MS = 10 * 1000;

export const useCameraMonitor = (cameras, capacity = DEFAULT_AREA_CAPACITY) => {
  const [counts, setCounts] = useState({}); // { cameraId: { count, timestamp } }
  const [globalAlert, setGlobalAlert] = useState(null); // { totalCount, occupancyPercent, alertLevel }
  const [alerts, setAlerts] = useState([]); // active non-safe alerts
  const [lastUpdated, setLastUpdated] = useState(null);
  const intervalRef = useRef(null);

  const fetchCounts = useCallback(async () => {
    if (!cameras || cameras.length === 0) return;

    try {
      let countData = [];

      // Try bulk endpoint first, fall back to individual calls
      try {
        countData = await cameraApi.getAllCameraCounts();
        console.log("✅ Fetched all camera counts:", countData);
      } catch (err) {
        console.log("Bulk fetch failed, trying individual:", err);
        const results = await Promise.all(
          cameras.map((cam) =>
            cameraApi
              .getCameraCount(cam.id)
              .then((data) => {
                console.log(`✅ Got count for ${cam.id}:`, data);
                return data;
              })
              .catch((e) => {
                console.log(`❌ Failed to get count for ${cam.id}:`, e);
                return { camera_id: cam.id, count: null, timestamp: null };
              }),
          ),
        );
        countData = results;
      }

      const newCounts = {};
      let totalCount = 0;
      const newAlerts = [];

      // Collect individual counts
      countData.forEach(({ camera_id, count, timestamp }) => {
        if (count === null || count === undefined) return;
        newCounts[camera_id] = { count, timestamp };
        totalCount += count;
      });

      // Calculate global alert based on total occupancy
      const globalAlertLevel = getAlertLevel(totalCount, capacity);
      const occupancyPercent = Math.round((totalCount / capacity) * 100);

      setGlobalAlert({
        totalCount,
        occupancyPercent,
        alertLevel: globalAlertLevel,
        timestamp: new Date().toISOString(),
      });

      // Add alert if global level is not SAFE
      if (globalAlertLevel.level !== "SAFE") {
        newAlerts.push({
          type: "AREA_WIDE",
          message: `Area occupancy at ${occupancyPercent}% (${totalCount}/${capacity} people)`,
          alertLevel: globalAlertLevel,
          timestamp: new Date().toISOString(),
        });
      }

      setCounts(newCounts);
      setAlerts(newAlerts);
      setLastUpdated(new Date());
    } catch (err) {
      console.error("Failed to fetch camera counts:", err);
    }
  }, [cameras, capacity]);

  useEffect(() => {
    fetchCounts();
    intervalRef.current = setInterval(fetchCounts, POLL_INTERVAL_MS);
    return () => clearInterval(intervalRef.current);
  }, [fetchCounts]);

  return {
    counts,
    globalAlert,
    alerts,
    lastUpdated,
    forceRefresh: fetchCounts,
  };
};
