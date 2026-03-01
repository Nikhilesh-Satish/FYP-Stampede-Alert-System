import styles from './CameraCard.module.css'

const CameraCard = ({ camera, countData, onDelete }) => {
  const count      = countData?.count ?? null
  const alertLevel = countData?.alertLevel
  const timestamp  = countData?.timestamp

  const fmtTime = (ts) => {
    if (!ts) return 'Pending first update…'
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div
      className={styles.card}
      style={{
        borderColor: alertLevel ? `${alertLevel.color}44` : 'rgba(255,255,255,0.07)',
        background:  alertLevel?.level !== 'SAFE' ? alertLevel.bg : '#1a2535',
      }}
    >
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.camIcon}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M23 7l-7 5 7 5V7z"/>
            <rect x="1" y="5" width="15" height="14" rx="2"/>
          </svg>
        </div>
        <div className={styles.camMeta}>
          <h3 className={styles.camName}>{camera.name}</h3>
          <p className={styles.camPath}>{camera.stream_path || camera.streamPath}</p>
        </div>
        <button className={styles.delBtn} onClick={() => onDelete(camera.id)} title="Remove">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14H6L5 6"/>
            <path d="M10 11v6M14 11v6M9 6V4h6v2"/>
          </svg>
        </button>
      </div>

      {/* Count */}
      <div className={styles.countRow}>
        {count !== null ? (
          <>
            <div className={styles.countDisplay}>
              <span className={styles.countNum} style={{ color: alertLevel?.color || '#00e5cc' }}>
                {count.toLocaleString()}
              </span>
              <span className={styles.countUnit}>people</span>
            </div>
            {alertLevel && (
              <div
                className={styles.badge}
                style={{
                  background:   alertLevel.bg,
                  color:        alertLevel.color,
                  borderColor:  `${alertLevel.color}44`,
                }}
              >
                {alertLevel.level === 'CRITICAL' && (
                  <span className={styles.blinkDot} style={{ background: alertLevel.color }} />
                )}
                {alertLevel.label}
              </div>
            )}
          </>
        ) : (
          <div className={styles.pending}>
            <div className={styles.spinner} />
            Awaiting data…
          </div>
        )}
      </div>

      {/* Footer */}
      <div className={styles.footer}>
        <span>Last update: {fmtTime(timestamp)}</span>
        <span className={styles.dot}>·</span>
        <span>Refreshes every 10 min</span>
      </div>
    </div>
  )
}

export default CameraCard
