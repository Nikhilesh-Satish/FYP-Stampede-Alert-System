import { useState, useEffect, useCallback, useRef } from "react";
import { cameraApi } from "../api/services";
import {
  getAlertLevel,
  getDensityAlertLevel,
  DEFAULT_AREA_CAPACITY,
} from "../api/config";
import { useChartData } from "../hooks/useChartData";

// Poll camera counts once per second for near real-time dashboard updates.
const POLL_INTERVAL_MS = 1000;

const buildHistoryPoint = (value, timestamp = new Date()) => {
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  return {
    time: date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    value,
    timestamp: date.getTime(),
  };
};

const appendHistoryPoint = (prev, point, maxPoints = 20) => {
  if (prev.length > 0 && prev[prev.length - 1].timestamp === point.timestamp) {
    const updated = [...prev];
    updated[updated.length - 1] = point;
    return updated;
  }
  return [...prev, point].slice(-maxPoints);
};

export const useCameraMonitor = (
  cameras,
  capacity = DEFAULT_AREA_CAPACITY,
  isPaused = true,
) => {
  const [counts, setCounts] = useState({}); // { cameraId: { count, timestamp } }
  const [globalAlert, setGlobalAlert] = useState(null); // { totalCount, occupancyPercent, alertLevel }
  const [alerts, setAlerts] = useState([]); // active non-safe alerts
  const [lastUpdated, setLastUpdated] = useState(null);
  const [densitySummary, setDensitySummary] = useState(null);
  const [globalDensityHistory, setGlobalDensityHistory] = useState([]);
  const intervalRef = useRef(null);
  const lastGlobalDensityTimestampRef = useRef(0);

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
      let latestTimestampMs = 0;
      let totalWeightedPeople = 0;
      let totalDensityArea = 0;
      let highestDensityScore = 0;
      let highestDensityCameraId = null;

      // Collect individual counts
      countData.forEach(({
        camera_id,
        count,
        in_count,
        out_count,
        timestamp,
        counting_enabled,
        density_enabled,
        weighted_people,
        density_score,
        smoothed_density_score,
        monitored_area_sqm,
        density_updated_at,
      }) => {
        if (count === null || count === undefined) return;

        const safeCount = Number.isFinite(Number(count)) ? Number(count) : 0;
        const pointTimestamp =
          typeof timestamp === "number" || typeof timestamp === "string"
            ? new Date(timestamp)
            : new Date();
        const pointTimestampMs = pointTimestamp.getTime();
        const cameraAlertLevel = getAlertLevel(Math.max(0, safeCount), capacity);
        const densityEnabled = Boolean(density_enabled);
        const smoothedDensity = Number(smoothed_density_score ?? 0);
        const densityAlertLevel = densityEnabled
          ? getDensityAlertLevel(smoothedDensity)
          : null;

        newCounts[camera_id] = {
          count: safeCount,
          inCount: in_count ?? 0,
          outCount: out_count ?? 0,
          timestamp,
          alertLevel: cameraAlertLevel,
          countingEnabled: counting_enabled ?? true,
          densityEnabled,
          weightedPeople: Number(weighted_people ?? 0),
          densityScore: Number(density_score ?? 0),
          smoothedDensityScore: smoothedDensity,
          monitoredAreaSqm: monitored_area_sqm ?? null,
          densityAlertLevel,
        };
        totalCount += safeCount;
        latestTimestampMs = Math.max(latestTimestampMs, pointTimestampMs);

        // Add data point to camera history (only if not paused)
        if (!isPaused) {
          addCameraDataPoint(camera_id, safeCount, pointTimestamp);
        }

        if (densityEnabled && densityAlertLevel && densityAlertLevel.level !== "SAFE") {
          const densityValue = smoothedDensity.toFixed(2);
          const weightedVisible = Number(weighted_people ?? 0).toFixed(1);
          const areaLabel = monitored_area_sqm ? `${monitored_area_sqm} m²` : "configured area";
          newAlerts.push({
            type: "DENSITY",
            message:
              densityAlertLevel.level === "CRITICAL"
                ? `Stampede risk on ${camera_id}: density score ${densityValue} across ${areaLabel} (${weightedVisible} weighted people visible)`
                : `High crowd density on ${camera_id}: density score ${densityValue} across ${areaLabel}`,
            alertLevel: densityAlertLevel,
            timestamp: pointTimestamp.toISOString(),
          });
        }

        if (densityEnabled && Number(monitored_area_sqm) > 0) {
          totalWeightedPeople += Number(weighted_people ?? 0);
          totalDensityArea += Number(monitored_area_sqm);
          latestTimestampMs = Math.max(
            latestTimestampMs,
            Number(density_updated_at || pointTimestampMs),
          );
          if (smoothedDensity > highestDensityScore) {
            highestDensityScore = smoothedDensity;
            highestDensityCameraId = camera_id;
          }
        }
      });

      const clampedTotalCount = Math.max(0, totalCount);
      const globalPointTimestamp = latestTimestampMs > 0 ? new Date(latestTimestampMs) : new Date();
      const globalDensityScore =
        totalDensityArea > 0 ? totalWeightedPeople / totalDensityArea : 0;
      const globalDensityAlertLevel = getDensityAlertLevel(globalDensityScore);

      // Add global data point (only if not paused)
      if (!isPaused) {
        addGlobalDataPoint(clampedTotalCount, globalPointTimestamp);
        const densityTimestampMs = countData.reduce((latest, entry) => {
          const value = Number(entry.density_updated_at || 0);
          return Number.isFinite(value) ? Math.max(latest, value) : latest;
        }, 0);
        if (
          densityTimestampMs > 0
          && densityTimestampMs !== lastGlobalDensityTimestampRef.current
        ) {
          lastGlobalDensityTimestampRef.current = densityTimestampMs;
          setGlobalDensityHistory((prev) =>
            appendHistoryPoint(prev, buildHistoryPoint(globalDensityScore, densityTimestampMs), 40),
          );
        }
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
      setDensitySummary({
        densityScore: globalDensityScore,
        weightedPeople: totalWeightedPeople,
        monitoredAreaSqm: totalDensityArea,
        highestCameraId: highestDensityCameraId,
        alertLevel: globalDensityAlertLevel,
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

      if (globalDensityAlertLevel.level !== "SAFE" && totalDensityArea > 0) {
        newAlerts.push({
          type: "STAMPede",
          message:
            globalDensityAlertLevel.level === "CRITICAL"
              ? `Approximate stampede risk: aggregate density score ${globalDensityScore.toFixed(2)} across monitored density cameras`
              : `Approximate crowd density elevated: aggregate density score ${globalDensityScore.toFixed(2)} across monitored density cameras`,
          alertLevel: globalDensityAlertLevel,
          timestamp: globalPointTimestamp.toISOString(),
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
    densitySummary,
    globalDensityHistory,
  };
};
