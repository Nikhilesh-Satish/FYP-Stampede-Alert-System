import styles from './AlertBanner.module.css'

const AlertBanner = ({ alerts }) => {
  if (!alerts || alerts.length === 0) return null

  const top = alerts.reduce((prev, cur) => {
    const order = { CRITICAL: 4, DANGER: 3, WARNING: 2, SAFE: 1 }
    return (order[cur.alertLevel.level] || 0) > (order[prev.alertLevel.level] || 0) ? cur : prev
  }, alerts[0])

  return (
    <div
      className={styles.banner}
      style={{ background: top.alertLevel.bg, borderColor: `${top.alertLevel.color}44` }}
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill={top.alertLevel.color}>
        <path d="M12 2L1 21h22L12 2zm0 3.5L20.5 19h-17L12 5.5zM11 10v4h2v-4h-2zm0 6v2h2v-2h-2z"/>
      </svg>
      <span className={styles.level} style={{ color: top.alertLevel.color }}>
        {top.alertLevel.label} Alert
      </span>
      <span className={styles.message}>
        {top.cameraName}: <strong>{top.count?.toLocaleString()}</strong> people detected
        {alerts.length > 1 && ` · ${alerts.length - 1} more camera${alerts.length > 2 ? 's' : ''} flagged`}
      </span>
    </div>
  )
}

export default AlertBanner
