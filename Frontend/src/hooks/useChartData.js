import { useState, useCallback, useEffect } from "react";

const normalizeTimestamp = (timestamp) => {
  if (timestamp instanceof Date) return timestamp;
  if (typeof timestamp === "number" || typeof timestamp === "string") {
    const date = new Date(timestamp);
    if (!Number.isNaN(date.getTime())) return date;
  }
  return new Date();
};

const makePoint = (count, timestamp) => {
  const date = normalizeTimestamp(timestamp);
  return {
    time: date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    count,
    timestamp: date.getTime(),
  };
};

// Hook to track historical data for charts
export const useChartData = (maxDataPoints = 20) => {
  const [globalHistory, setGlobalHistory] = useState([]);
  const [cameraHistory, setCameraHistory] = useState({}); // { cameraId: [...history] }

  const addGlobalDataPoint = useCallback(
    (count, timestamp = new Date()) => {
      setGlobalHistory((prev) => {
        const point = makePoint(count, timestamp);
        if (prev.length > 0 && prev[prev.length - 1].timestamp === point.timestamp) {
          const updated = [...prev];
          updated[updated.length - 1] = point;
          return updated;
        }
        const newData = [...prev, point];
        // Keep only last maxDataPoints
        return newData.slice(-maxDataPoints);
      });
    },
    [maxDataPoints],
  );

  const addCameraDataPoint = useCallback(
    (cameraId, count, timestamp = new Date()) => {
      setCameraHistory((prev) => {
        const camHistory = prev[cameraId] || [];
        const point = makePoint(count, timestamp);
        const newData =
          camHistory.length > 0 && camHistory[camHistory.length - 1].timestamp === point.timestamp
            ? [...camHistory.slice(0, -1), point]
            : [...camHistory, point];
        return {
          ...prev,
          [cameraId]: newData.slice(-maxDataPoints),
        };
      });
    },
    [maxDataPoints],
  );

  const clearHistory = useCallback(() => {
    setGlobalHistory([]);
    setCameraHistory({});
  }, []);

  return {
    globalHistory,
    cameraHistory,
    addGlobalDataPoint,
    addCameraDataPoint,
    clearHistory,
  };
};
