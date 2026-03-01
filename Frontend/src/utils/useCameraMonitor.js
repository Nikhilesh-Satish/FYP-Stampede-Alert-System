import { useState, useEffect, useCallback, useRef } from 'react'
import { cameraApi } from '../api/services'
import { getAlertLevel } from '../api/config'

// Polls camera counts from your API every 10 minutes
const POLL_INTERVAL_MS = 10 * 60 * 1000

export const useCameraMonitor = (cameras) => {
  const [counts,      setCounts]      = useState({})  // { cameraId: { count, timestamp, alertLevel } }
  const [alerts,      setAlerts]      = useState([])  // active non-safe alerts
  const [lastUpdated, setLastUpdated] = useState(null)
  const intervalRef = useRef(null)

  const fetchCounts = useCallback(async () => {
    if (!cameras || cameras.length === 0) return

    try {
      let countData = []

      // Try bulk endpoint first, fall back to individual calls
      try {
        countData = await cameraApi.getAllCameraCounts()
      } catch {
        const results = await Promise.all(
          cameras.map((cam) =>
            cameraApi.getCameraCount(cam.id)
              .catch(() => ({ camera_id: cam.id, count: null, timestamp: null }))
          )
        )
        countData = results
      }

      const newCounts = {}
      const newAlerts = []

      countData.forEach(({ camera_id, count, timestamp }) => {
        if (count === null || count === undefined) return
        const alertLevel = getAlertLevel(count)
        newCounts[camera_id] = { count, timestamp, alertLevel }

        if (alertLevel.level !== 'SAFE') {
          const cam = cameras.find((c) => c.id === camera_id)
          newAlerts.push({
            cameraId:   camera_id,
            cameraName: cam?.name || camera_id,
            count,
            alertLevel,
            timestamp:  timestamp || new Date().toISOString(),
          })
        }
      })

      setCounts(newCounts)
      setAlerts(newAlerts)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Failed to fetch camera counts:', err)
    }
  }, [cameras])

  useEffect(() => {
    fetchCounts()
    intervalRef.current = setInterval(fetchCounts, POLL_INTERVAL_MS)
    return () => clearInterval(intervalRef.current)
  }, [fetchCounts])

  return { counts, alerts, lastUpdated, forceRefresh: fetchCounts }
}
