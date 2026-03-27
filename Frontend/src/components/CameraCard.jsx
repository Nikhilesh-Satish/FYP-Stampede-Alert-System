import { useState } from "react";
import { cameraApi } from "../api/services";
import styles from "./CameraCard.module.css";
import { usePointerGlow } from "../hooks/usePointerGlow";

const CameraCard = ({ camera, countData, onDelete, isPaused = false }) => {
  const glow = usePointerGlow();
  const [resetLoading, setResetLoading] = useState(false);
  const [isIndividuallyPaused, setIsIndividuallyPaused] = useState(false);
  const [startStopLoading, setStartStopLoading] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [isZoomOpen, setIsZoomOpen] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const count = countData?.count ?? null;
  const inCount = countData?.inCount ?? 0;
  const outCount = countData?.outCount ?? 0;
  const alertLevel = countData?.alertLevel;
  const timestamp = countData?.timestamp;
  const streamUrl = cameraApi.getCameraStreamUrl(camera?.id, {
    overlay: showOverlay,
  });

  const fmtTime = (ts) => {
    if (!ts) return "Pending first update…";
    return new Date(ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const handleResetCount = async () => {
    if (!window.confirm("Reset count to 0 for this camera?")) return;

    setResetLoading(true);
    try {
      await cameraApi.resetCameraCount(camera.id);
      alert("Count reset successfully!");
    } catch (err) {
      alert("Failed to reset count: " + err.message);
    } finally {
      setResetLoading(false);
    }
  };

  const handleIndividualPauseToggle = async () => {
    if (isPaused) return;

    setStartStopLoading(true);
    try {
      if (isIndividuallyPaused) {
        await cameraApi.startCamera(camera.id);
        setIsIndividuallyPaused(false);
      } else {
        await cameraApi.stopCamera(camera.id);
        setIsIndividuallyPaused(true);
      }
    } catch (err) {
      alert(
        `Failed to ${isIndividuallyPaused ? "start" : "stop"} camera: ` + err.message,
      );
    } finally {
      setStartStopLoading(false);
    }
  };

  return (
    <>
      <div
        ref={glow.ref}
        {...glow.glowProps}
        className={styles.card}
        style={{
          borderColor: alertLevel ? `${alertLevel.color}40` : "var(--line)",
          background:
            alertLevel?.level !== "SAFE" && alertLevel
              ? `linear-gradient(180deg, ${alertLevel.bg}, rgba(8,16,31,0.98))`
              : undefined,
        }}
      >
        <div className={styles.header}>
          <div className={styles.cameraSignature}>
            <div className={styles.camIcon}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
                <path d="M23 7l-7 5 7 5V7z" />
                <rect x="1" y="5" width="15" height="14" rx="2" />
              </svg>
            </div>
            <div className={styles.camMeta}>
              <h3 className={styles.camName}>{camera?.name || "Unnamed Camera"}</h3>
              <p className={styles.camPath}>
                {camera?.stream_path || camera?.streamPath || "No path specified"}
              </p>
            </div>
          </div>

          <div className={styles.headerActions}>
            <button
              className={`${styles.individualPauseBtn} ${
                isIndividuallyPaused || isPaused ? styles.individualPaused : styles.individualActive
              }`}
              onClick={handleIndividualPauseToggle}
              disabled={startStopLoading || isPaused}
              title={
                isPaused
                  ? "Global monitoring is paused"
                  : isIndividuallyPaused
                    ? "Resume this camera"
                    : "Pause this camera"
              }
            >
              {startStopLoading ? (
                <span className={styles.spinner} />
              ) : isIndividuallyPaused || isPaused ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              )}
            </button>

            <button className={styles.delBtn} onClick={() => onDelete(camera?.id)} title="Remove">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-1 14H6L5 6" />
                <path d="M10 11v6M14 11v6M9 6V4h6v2" />
              </svg>
            </button>
          </div>
        </div>

        <div className={styles.streamWrap}>
          <div className={styles.streamControls}>
            <button type="button" className={styles.streamToggle} onClick={() => setShowOverlay((prev) => !prev)}>
              {showOverlay ? "Show Raw Feed" : "Show Algorithm Overlay"}
            </button>
            <button
              type="button"
              className={styles.streamZoomBtn}
              onClick={() => setIsZoomOpen(true)}
              disabled={streamError}
            >
              Magnify
            </button>
          </div>

          <button
            type="button"
            className={styles.streamPreview}
            onClick={() => setIsZoomOpen(true)}
            disabled={streamError}
          >
            <img
              src={streamUrl}
              alt={`${camera?.name || "Camera"} live stream`}
              onError={() => setStreamError(true)}
              onLoad={() => setStreamError(false)}
            />
            {streamError && <span className={styles.streamError}>Stream unavailable</span>}
          </button>

          <p className={styles.streamLegend}>
            {showOverlay
              ? "Overlay: green boxes with tracker IDs. Rule: left-to-right = IN, right-to-left = OUT."
              : "Raw live feed without algorithm annotations."}
          </p>
        </div>

        <div className={styles.countRow}>
          {isPaused || isIndividuallyPaused ? (
            <div className={styles.paused}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
              <span>{isPaused ? "Monitoring Paused" : "Camera Paused"}</span>
            </div>
          ) : count !== null ? (
            <>
              <div className={styles.countSummary}>
                <div className={styles.countMetric}>
                  <span className={styles.countLabel}>IN</span>
                  <span className={styles.countValueIn}>{inCount.toLocaleString()}</span>
                </div>
                <div className={styles.countMetric}>
                  <span className={styles.countLabel}>OUT</span>
                  <span className={styles.countValueOut}>{outCount.toLocaleString()}</span>
                </div>
              </div>

              <div className={styles.countControls}>
                {alertLevel && (
                  <div
                    className={styles.badge}
                    style={{
                      background: alertLevel.bg,
                      color: alertLevel.color,
                      borderColor: `${alertLevel.color}44`,
                    }}
                  >
                    {alertLevel.level === "CRITICAL" && <span className={styles.blinkDot} />}
                    {alertLevel.label}
                  </div>
                )}

                <button
                  className={styles.resetBtn}
                  onClick={handleResetCount}
                  disabled={resetLoading}
                  title="Reset camera count"
                >
                  {resetLoading ? (
                    <span className={styles.resetSpinner} />
                  ) : (
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="23 4 23 10 17 10" />
                      <polyline points="1 20 1 14 7 14" />
                      <path d="M3.5 9a9 9 0 0 1 14.8-3.3L23 10M1 14l4.7 4.3A9 9 0 0 0 20.5 15" />
                    </svg>
                  )}
                </button>
              </div>
            </>
          ) : (
            <div className={styles.pending}>
              <span className={styles.spinner} />
              Waiting for data…
            </div>
          )}
        </div>

        <div className={styles.footer}>
          <span>Current total: {count ?? "..."}</span>
          <span className={styles.dot}>•</span>
          <span>Last update: {fmtTime(timestamp)}</span>
        </div>
      </div>

      {isZoomOpen && (
        <div className={styles.modalOverlay} onClick={() => setIsZoomOpen(false)}>
          <div className={styles.modalBody} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h4>{camera?.name || "Camera"} Live Feed</h4>
              <button onClick={() => setIsZoomOpen(false)}>Close</button>
            </div>
            <div className={styles.modalStream}>
              <img src={streamUrl} alt={`${camera?.name || "Camera"} magnified stream`} />
            </div>
            <p className={styles.modalHint}>
              {showOverlay ? "Algorithm overlay active in zoom view." : "Raw live feed in zoom view."}
            </p>
          </div>
        </div>
      )}
    </>
  );
};

export default CameraCard;
