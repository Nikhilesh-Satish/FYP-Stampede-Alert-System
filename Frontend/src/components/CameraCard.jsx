import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { cameraApi } from "../api/services";
import styles from "./CameraCard.module.css";
import { usePointerGlow } from "../hooks/usePointerGlow";
import InfoButton from "./InfoButton";

const DENSITY_INFO_TEXT =
  "This is an estimate, not an exact physical measurement. We divide the camera view into three horizontal zones, give more importance to people farther away in ground-camera views, and then compare that weighted crowd size against the monitored area you entered.";

const CameraCard = ({ camera, countData, onDelete, isPaused = false }) => {
  const glow = usePointerGlow();
  const [resetLoading, setResetLoading] = useState(false);
  const [isIndividuallyPaused, setIsIndividuallyPaused] = useState(false);
  const [startStopLoading, setStartStopLoading] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [isZoomOpen, setIsZoomOpen] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const [zoomStreamError, setZoomStreamError] = useState(false);
  const count = countData?.count ?? null;
  const inCount = countData?.inCount ?? 0;
  const outCount = countData?.outCount ?? 0;
  const alertLevel = countData?.alertLevel;
  const countingEnabled = countData?.countingEnabled ?? camera?.counting_enabled ?? true;
  const densityEnabled = countData?.densityEnabled ?? camera?.density_enabled ?? false;
  const weightedPeople = countData?.weightedPeople ?? 0;
  const smoothedDensityScore = countData?.smoothedDensityScore ?? 0;
  const densityAlertLevel = countData?.densityAlertLevel;
  const monitoredAreaSqm = countData?.monitoredAreaSqm ?? camera?.monitored_area_sqm ?? null;
  const timestamp = countData?.timestamp;
  const streamUrl = cameraApi.getCameraStreamUrl(camera?.id, {
    overlay: showOverlay,
  });
  const directionHint = (() => {
    const axis = camera?.count_axis || "x";
    const inDirection = camera?.in_direction || "positive";
    if (axis === "y") {
      return inDirection === "positive"
        ? "top-to-bottom = IN, bottom-to-top = OUT"
        : "bottom-to-top = IN, top-to-bottom = OUT";
    }
    return inDirection === "positive"
      ? "left-to-right = IN, right-to-left = OUT"
      : "right-to-left = IN, left-to-right = OUT";
  })();
  const shotTypeLabel = camera?.shot_type === "drone" ? "Drone / overhead mode" : "Ground / normal mode";
  const processingPresetLabel = (() => {
    switch (camera?.processing_preset) {
      case "fast":
        return "Fast preset";
      case "accurate":
        return "Accurate preset";
      default:
        return "Balanced preset";
    }
  })();
  const highestAlert = (() => {
    const severity = { SAFE: 0, WARNING: 1, DANGER: 2, CRITICAL: 3 };
    const countSeverity = severity[alertLevel?.level] ?? 0;
    const densitySeverity = severity[densityAlertLevel?.level] ?? 0;
    return densitySeverity > countSeverity ? densityAlertLevel : alertLevel;
  })();

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

  useEffect(() => {
    if (!isZoomOpen) return undefined;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setIsZoomOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isZoomOpen]);

  const openZoom = () => {
    setZoomStreamError(false);
    setIsZoomOpen(true);
  };

  return (
    <>
      <div
        ref={glow.ref}
        {...glow.glowProps}
        className={styles.card}
        style={{
          borderColor: highestAlert ? `${highestAlert.color}40` : "var(--line)",
          background:
            highestAlert?.level !== "SAFE" && highestAlert
              ? `linear-gradient(180deg, ${highestAlert.bg}, rgba(8,16,31,0.98))`
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
            <button type="button" className={styles.streamZoomBtn} onClick={openZoom}>
              Magnify
            </button>
          </div>

          <button
            type="button"
            className={styles.streamPreview}
            onClick={openZoom}
            disabled={streamError}
            style={{
              borderColor: highestAlert ? `${highestAlert.color}44` : undefined,
              boxShadow:
                highestAlert && highestAlert.level !== "SAFE"
                  ? `0 0 0 1px ${highestAlert.color}22, 0 18px 38px ${highestAlert.color}18`
                  : undefined,
            }}
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
              ? `Overlay: green boxes with tracker IDs. ${shotTypeLabel}. ${processingPresetLabel}. Rule: ${directionHint}.`
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
              {countingEnabled ? (
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
              ) : densityEnabled ? (
                <div className={styles.countSummary}>
                  <div className={styles.countMetric}>
                    <span className={styles.countLabel}>
                      Approx. Density
                      <InfoButton
                        text={DENSITY_INFO_TEXT}
                        label="Crowd density methodology"
                      />
                    </span>
                    <span className={styles.countValueIn}>{smoothedDensityScore.toFixed(2)}</span>
                  </div>
                  <div className={styles.countMetric}>
                    <span className={styles.countLabel}>Weighted People</span>
                    <span className={styles.countValueOut}>{weightedPeople.toFixed(1)}</span>
                  </div>
                </div>
              ) : null}

              <div className={styles.countControls}>
                {countingEnabled && alertLevel && (
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
                {densityEnabled && densityAlertLevel && (
                  <div
                    className={styles.badge}
                    style={{
                      background: densityAlertLevel.bg,
                      color: densityAlertLevel.color,
                      borderColor: `${densityAlertLevel.color}44`,
                    }}
                  >
                    {densityAlertLevel.level === "CRITICAL" && <span className={styles.blinkDot} />}
                    Density {densityAlertLevel.label}
                  </div>
                )}

                {countingEnabled && (
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
                )}
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
          <span>
            {countingEnabled
              ? `Current total: ${count ?? "..."}`
              : `Current total: ${count ?? "..."}`}
          </span>
          <span className={styles.dot}>•</span>
          <span>Last update: {fmtTime(timestamp)}</span>
        </div>

        {densityEnabled && (
          <div className={styles.densityPanel}>
            <div className={styles.densityHeader}>
              <span className={styles.densityTitle}>Approximate Crowd Density</span>
              <InfoButton
                text={DENSITY_INFO_TEXT}
                label="Crowd density methodology"
              />
            </div>
            <div className={styles.densityGrid}>
              <div className={styles.densityItem}>
                <span className={styles.densityLabel}>Density Score</span>
                <strong className={styles.densityValue}>{smoothedDensityScore.toFixed(2)}</strong>
              </div>
              <div className={styles.densityItem}>
                <span className={styles.densityLabel}>Weighted People</span>
                <strong className={styles.densityValue}>{weightedPeople.toFixed(1)}</strong>
              </div>
              <div className={styles.densityItem}>
                <span className={styles.densityLabel}>Area</span>
                <strong className={styles.densityValue}>
                  {monitoredAreaSqm ? `${monitoredAreaSqm} m²` : "Not set"}
                </strong>
              </div>
            </div>
          </div>
        )}
      </div>

      {isZoomOpen &&
        createPortal(
          <div
            className={styles.modalOverlay}
            onClick={() => setIsZoomOpen(false)}
            role="dialog"
            aria-modal="true"
            aria-label={`${camera?.name || "Camera"} live feed`}
          >
            <div className={styles.modalBody} onClick={(e) => e.stopPropagation()}>
              <div
                className={styles.modalHeader}
                style={{
                  borderColor: highestAlert ? `${highestAlert.color}44` : undefined,
                  background: highestAlert?.bg,
                }}
              >
                <h4>{camera?.name || "Camera"} Live Feed</h4>
                <button type="button" onClick={() => setIsZoomOpen(false)}>
                  Close
                </button>
              </div>
              <div className={styles.modalStream}>
                {zoomStreamError ? (
                  <div className={styles.modalError}>Unable to load the magnified stream.</div>
                ) : (
                  <img
                    src={streamUrl}
                    alt={`${camera?.name || "Camera"} magnified stream`}
                    onError={() => setZoomStreamError(true)}
                  />
                )}
              </div>
              <p className={styles.modalHint}>
                {showOverlay ? "Algorithm overlay active in zoom view." : "Raw live feed in zoom view."}
              </p>
            </div>
          </div>
          ,
          document.body,
        )}
    </>
  );
};

export default CameraCard;
