import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { cameraApi } from '../api/services'
import { useCameraMonitor } from '../utils/useCameraMonitor'
import Navbar from '../components/Navbar'
import CameraCard from '../components/CameraCard'
import AddCameraModal from '../components/AddCameraModal'
import AlertBanner from '../components/AlertBanner'
import styles from './DashboardPage.module.css'

// Small stat card
const StatCard = ({ label, value, color, icon }) => (
  <div className={styles.statCard}>
    <div className={styles.statIcon} style={{ color }}>{icon}</div>
    <div>
      <p className={styles.statValue} style={{ color }}>{value}</p>
      <p className={styles.statLabel}>{label}</p>
    </div>
  </div>
)

const DashboardPage = () => {
  const { user } = useAuth()
  const [cameras,       setCameras]       = useState([])
  const [loadingCams,   setLoadingCams]   = useState(true)
  const [fetchError,    setFetchError]    = useState('')
  const [showAddModal,  setShowAddModal]  = useState(false)

  const { counts, alerts, lastUpdated, forceRefresh } = useCameraMonitor(cameras)

  // Load cameras on mount
  useEffect(() => {
    cameraApi.getCameras()
      .then(data => setCameras(Array.isArray(data) ? data : (data.cameras || [])))
      .catch(() => setFetchError('Could not load cameras. Is your backend running?'))
      .finally(() => setLoadingCams(false))
  }, [])

  const handleCameraAdded = (cam) => {
    setCameras(prev => [...prev, cam])
    setTimeout(forceRefresh, 500)
  }

  const handleDelete = async (id) => {
    if (!window.confirm('Remove this camera from monitoring?')) return
    try {
      await cameraApi.deleteCamera(id)
      setCameras(prev => prev.filter(c => c.id !== id))
    } catch (err) {
      alert('Delete failed: ' + err.message)
    }
  }

  // Aggregate stats
  const totalPeople   = Object.values(counts).reduce((s, d) => s + (d.count || 0), 0)
  const criticalCount = alerts.filter(a => a.alertLevel.level === 'CRITICAL').length
  const dangerCount   = alerts.filter(a => a.alertLevel.level === 'DANGER').length
  const warningCount  = alerts.filter(a => a.alertLevel.level === 'WARNING').length

  return (
    <div className={styles.page}>
      <Navbar alertCount={alerts.length} />

      <main className={styles.main}>
        {/* Page header */}
        <div className={styles.pageHeader}>
          <div>
            <h1 className={styles.pageTitle}>Admin Control Panel</h1>
            <p className={styles.pageSubtitle}>
              Welcome back, <span className={styles.hi}>{user?.firstName || 'Admin'}</span>
              {lastUpdated && (
                <span className={styles.ts}>
                  {' '}· Updated {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              )}
            </p>
          </div>
          <div className={styles.headerBtns}>
            <button className={styles.refreshBtn} onClick={forceRefresh} title="Refresh now">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 4 23 10 17 10"/>
                <polyline points="1 20 1 14 7 14"/>
                <path d="M3.5 9a9 9 0 0 1 14.8-3.3L23 10M1 14l4.7 4.3A9 9 0 0 0 20.5 15"/>
              </svg>
            </button>
            <button className={styles.addBtn} onClick={() => setShowAddModal(true)}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="5" x2="12" y2="19"/>
                <line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              Add Camera
            </button>
          </div>
        </div>

        {/* Alert banner */}
        <AlertBanner alerts={alerts} />

        {/* Stats */}
        <div className={styles.statsRow}>
          <StatCard label="Total People" value={totalPeople.toLocaleString()} color="#00e5cc"
            icon={<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="9" cy="7" r="3"/><path d="M3 21v-2a6 6 0 0 1 6-6"/><circle cx="17" cy="7" r="3"/><path d="M21 21v-2a6 6 0 0 0-6-6"/></svg>}
          />
          <StatCard label="Active Cameras" value={cameras.length} color="#7dd3fc"
            icon={<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>}
          />
          <StatCard label="Active Alerts" value={alerts.length} color={alerts.length > 0 ? '#ef4444' : '#22c55e'}
            icon={<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M10.3 21h3.4M12 3a7 7 0 0 1 7 7c0 2.5 1 4 1 6H4c0-2 1-3.5 1-6a7 7 0 0 1 7-7z"/></svg>}
          />
          <StatCard label="Critical / Danger / Warning" value={`${criticalCount} / ${dangerCount} / ${warningCount}`} color="#f59e0b"
            icon={<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 2L1 21h22L12 2z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>}
          />
        </div>

        {/* Error */}
        {fetchError && <div className={styles.errorBox}>{fetchError}</div>}

        {/* Camera grid */}
        {loadingCams ? (
          <div className={styles.loadingState}>
            <div className={styles.spinner} />
            Loading cameras…
          </div>
        ) : cameras.length === 0 ? (
          <div className={styles.emptyState}>
            <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#1e2a3a" strokeWidth="0.8">
              <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
            </svg>
            <h3 className={styles.emptyTitle}>No cameras yet</h3>
            <p className={styles.emptyText}>Register your first camera to start monitoring crowd density.</p>
            <button className={styles.addBtn} onClick={() => setShowAddModal(true)}>Add First Camera</button>
          </div>
        ) : (
          <div className={styles.grid}>
            {cameras.map(cam => (
              <CameraCard
                key={cam.id}
                camera={cam}
                countData={counts[cam.id]}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}
      </main>

      {showAddModal && (
        <AddCameraModal onClose={() => setShowAddModal(false)} onSuccess={handleCameraAdded} />
      )}
    </div>
  )
}

export default DashboardPage
