import { useState, useCallback, useEffect } from "react";

// Hook to track historical data for charts
export const useChartData = (maxDataPoints = 20) => {
  const [globalHistory, setGlobalHistory] = useState([]);
  const [cameraHistory, setCameraHistory] = useState({}); // { cameraId: [...history] }

  const addGlobalDataPoint = useCallback(
    (count, timestamp = new Date()) => {
      setGlobalHistory((prev) => {
        const newData = [
          ...prev,
          {
            time: timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
            count,
            timestamp: timestamp.getTime(),
          },
        ];
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
        const newData = [
          ...camHistory,
          {
            time: timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
            count,
            timestamp: timestamp.getTime(),
          },
        ];
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
