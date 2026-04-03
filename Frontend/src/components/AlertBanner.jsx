import styles from "./AlertBanner.module.css";

const AlertBanner = ({ alerts }) => {
  if (!alerts || alerts.length === 0) return null;

  const order = { CRITICAL: 4, DANGER: 3, WARNING: 2, SAFE: 1 };
  const sortedAlerts = [...alerts].sort(
    (a, b) => (order[b.alertLevel.level] || 0) - (order[a.alertLevel.level] || 0),
  );

  return (
    <div className={styles.stack}>
      {sortedAlerts.map((alert, index) => (
        <div
          key={`${alert.type || "alert"}-${alert.timestamp || index}-${index}`}
          className={`${styles.banner} ${styles[alert.alertLevel.level?.toLowerCase()] || ""}`}
          style={{
            background: alert.alertLevel.bg,
            borderColor: `${alert.alertLevel.color}44`,
          }}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill={alert.alertLevel.color}
          >
            <path d="M12 2L1 21h22L12 2zm0 3.5L20.5 19h-17L12 5.5zM11 10v4h2v-4h-2zm0 6v2h2v-2h-2z" />
          </svg>
          <span className={styles.level} style={{ color: alert.alertLevel.color }}>
            {alert.alertLevel.label} Alert
          </span>
          <span className={styles.message}>{alert.message}</span>
        </div>
      ))}
    </div>
  );
};

export default AlertBanner;
