import { useState, useEffect, useCallback, useRef } from "react";
import { cameraApi } from "../api/services";
import { getAlertLevel, DEFAULT_AREA_CAPACITY } from "../api/config";
import { useChartData } from "../hooks/useChartData";

// Poll camera counts once per second for near real-time dashboard updates.
const POLL_INTERVAL_MS = 1000;

export const useCameraMonitor = (
  cameras,
  capacity = DEFAULT_AREA_CAPACITY,
  isPaused = true,
) => {
  const [counts, setCounts] = useState({}); // { cameraId: { count, timestamp } }
  const [globalAlert, setGlobalAlert] = useState(null); // { totalCount, occupancyPercent, alertLevel }
  const [alerts, setAlerts] = useState([]); // active non-safe alerts
  const [lastUpdated, setLastUpdated] = useState(null);
  const intervalRef = useRef(null);

  // Use chart data hook for historical tracking
  const {
    globalHistory,
    cameraHistory,
    addGlobalDataPoint,
    addCameraDataPoint,
  } = useChartData();

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
      countData.forEach(({ camera_id, count, in_count, out_count, timestamp }) => {
        if (count === null || count === undefined) return;

        const safeCount = Number.isFinite(Number(count)) ? Number(count) : 0;
        const cameraAlertLevel = getAlertLevel(Math.max(0, safeCount), capacity);

        newCounts[camera_id] = {
          count: safeCount,
          inCount: in_count ?? 0,
          outCount: out_count ?? 0,
          timestamp,
          alertLevel: cameraAlertLevel,
        };
        totalCount += safeCount;

        // Add data point to camera history (only if not paused)
        if (!isPaused) {
          addCameraDataPoint(camera_id, count);
        }
      });

      const clampedTotalCount = Math.max(0, totalCount);

      // Add global data point (only if not paused)
      if (!isPaused) {
        addGlobalDataPoint(clampedTotalCount);
      }

      // Calculate global alert based on total occupancy
      const globalAlertLevel = getAlertLevel(clampedTotalCount, capacity);
      const occupancyPercent = Math.round((clampedTotalCount / capacity) * 100);

      setGlobalAlert({
        totalCount: clampedTotalCount,
        occupancyPercent,
        alertLevel: globalAlertLevel,
        timestamp: new Date().toISOString(),
      });

      // Add alert if global level is not SAFE
      if (globalAlertLevel.level !== "SAFE") {
        let alertMessage = "";

        if (clampedTotalCount > capacity) {
          const exceeded = clampedTotalCount - capacity;
          alertMessage = `🚨 CAPACITY EXCEEDED: ${exceeded} people over safe limit (${clampedTotalCount}/${capacity})`;
        } else if (occupancyPercent >= 80) {
          alertMessage = `⚠️ Approaching safe capacity: ${clampedTotalCount}/${capacity} people (${occupancyPercent}%)`;
        } else {
          alertMessage = `Area occupancy at ${occupancyPercent}% (${clampedTotalCount}/${capacity} people)`;
        }

        newAlerts.push({
          type: "AREA_WIDE",
          message: alertMessage,
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
  }, [cameras, capacity, isPaused, addGlobalDataPoint, addCameraDataPoint]);

  // Function to restart the interval
  const restartInterval = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(() => {
      fetchCounts();
    }, POLL_INTERVAL_MS);
  }, []);

  // Manual refresh: fetch immediately and restart interval
  const forceRefresh = useCallback(async () => {
    await fetchCounts();
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(() => {
      fetchCounts();
    }, POLL_INTERVAL_MS);
  }, [fetchCounts]);

  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Even in paused mode, fetch once so the dashboard shows the latest
    // aggregate count instead of staying empty until polling is resumed.
    fetchCounts();

    if (isPaused) {
      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }

    intervalRef.current = setInterval(() => {
      fetchCounts();
    }, POLL_INTERVAL_MS);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchCounts, isPaused]);

  return {
    counts,
    globalAlert,
    alerts,
    lastUpdated,
    forceRefresh,
    globalHistory,
    cameraHistory,
  };
};
