import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { cameraApi } from "../api/services";
import { useCameraMonitor } from "../utils/useCameraMonitor";
import { DEFAULT_AREA_CAPACITY } from "../api/config";
import Navbar from "../components/Navbar";
import CameraCard from "../components/CameraCard";
import AddCameraModal from "../components/AddCameraModal";
import AlertBanner from "../components/AlertBanner";
import GlobalTrendChart from "../components/GlobalTrendChart";
import CameraTrendChart from "../components/CameraTrendChart";
import PredictionCard from "../components/PredictionCard";
import styles from "./DashboardPage.module.css";

// Small stat card
const StatCard = ({ label, value, color, icon }) => (
  <div className={styles.statCard}>
    <div className={styles.statIcon} style={{ color }}>
      {icon}
    </div>
    <div>
      <p className={styles.statValue} style={{ color }}>
        {value}
      </p>
      <p className={styles.statLabel}>{label}</p>
    </div>
  </div>
);

const DashboardPage = () => {
  const { user } = useAuth();
  const [cameras, setCameras] = useState([]);
  const [loadingCams, setLoadingCams] = useState(true);
  const [fetchError, setFetchError] = useState("");
  const [showAddModal, setShowAddModal] = useState(false);
  const [capacity, setCapacity] = useState(DEFAULT_AREA_CAPACITY);
  const [resetLoading, setResetLoading] = useState(false);
  const [isPaused, setIsPaused] = useState(true);

  const {
    counts,
    globalAlert,
    alerts,
    lastUpdated,
    forceRefresh,
    globalHistory,
    cameraHistory,
  } = useCameraMonitor(cameras, capacity, isPaused);

  // Load cameras on mount
  useEffect(() => {
    cameraApi
      .getCameras()
      .then((data) => {
        console.log("Cameras loaded:", data);
        setCameras(Array.isArray(data) ? data : data.cameras || []);
      })
      .catch((err) => {
        console.error("Failed to load cameras:", err);
        setFetchError("Could not load cameras. Is your backend running?");
      })
      .finally(() => setLoadingCams(false));
  }, []);

  const handleCameraAdded = (cam) => {
    console.log("Camera added:", cam);
    setCameras((prev) => [...prev, cam]);
    setTimeout(forceRefresh, 500);
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Remove this camera from monitoring?")) return;
    try {
      await cameraApi.deleteCamera(id);
      setCameras((prev) => prev.filter((c) => c.id !== id));
    } catch (err) {
      alert("Delete failed: " + err.message);
    }
  };

  const handleResetGlobalCount = async () => {
    if (!window.confirm("Reset total count to 0 for all cameras?")) return;

    setResetLoading(true);
    try {
      await cameraApi.resetAllCameraCounts();
      forceRefresh();
      alert("Global count reset successfully!");
    } catch (err) {
      alert("Failed to reset count: " + err.message);
    } finally {
      setResetLoading(false);
    }
  };

  // Global stats from aggregate monitoring
  const totalPeople = globalAlert?.totalCount ?? 0;
  const occupancyPercent = globalAlert?.occupancyPercent ?? 0;
  const alertLevel = globalAlert?.alertLevel;

  return (
    <div className={styles.page}>
      <Navbar alertCount={alerts.length} />

      <main className={styles.main}>
        {/* Page header */}
        <div className={styles.pageHeader}>
          <div>
            <h1 className={styles.pageTitle}>Admin Control Panel</h1>
            <p className={styles.pageSubtitle}>
              Welcome back,{" "}
              <span className={styles.hi}>{user?.firstName || "Admin"}</span>
              {lastUpdated && (
                <span className={styles.ts}>
                  {" "}
                  · Updated{" "}
                  {lastUpdated.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              )}
            </p>
          </div>
          <div className={styles.headerBtns}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "10px 16px",
                backgroundColor: "rgba(15, 23, 42, 0.6)",
                borderRadius: "8px",
                border: "1px solid rgba(0, 229, 204, 0.2)",
              }}
            >
              <label
                style={{ fontSize: "13px", fontWeight: 600, color: "#cbd5e1" }}
              >
                Area Capacity:
              </label>
              <input
                type="number"
                min="1"
                value={capacity}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === "") {
                    setCapacity("");
                  } else {
                    const numVal = parseInt(val, 10);
                    if (numVal > 0) setCapacity(numVal);
                  }
                }}
                onBlur={(e) => {
                  if (
                    e.target.value === "" ||
                    parseInt(e.target.value, 10) < 1
                  ) {
                    setCapacity(DEFAULT_AREA_CAPACITY);
                  }
                }}
                style={{
                  width: "120px",
                  padding: "8px 12px",
                  border: "2px solid rgba(0, 229, 204, 0.3)",
                  borderRadius: "6px",
                  fontSize: "15px",
                  fontWeight: 600,
                  textAlign: "center",
                  backgroundColor: "rgba(0, 229, 204, 0.05)",
                  color: "#00e5cc",
                  outlineColor: "rgba(0, 229, 204, 0.5)",
                  transition: "all 0.2s",
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = "rgba(0, 229, 204, 0.6)";
                  e.target.style.backgroundColor = "rgba(0, 229, 204, 0.1)";
                  e.target.style.boxShadow = "0 0 8px rgba(0, 229, 204, 0.2)";
                }}
                onBlurCapture={(e) => {
                  e.target.style.borderColor = "rgba(0, 229, 204, 0.3)";
                  e.target.style.backgroundColor = "rgba(0, 229, 204, 0.05)";
                  e.target.style.boxShadow = "none";
                }}
              />
              <span style={{ fontSize: "13px", color: "#94a3b8" }}>people</span>
            </div>
            <button
              className={styles.playPauseBtn}
              onClick={() => setIsPaused(!isPaused)}
              disabled={false}
              title={
                isPaused
                  ? "Start monitoring all cameras"
                  : "Pause monitoring all cameras"
              }
              style={{
                background: isPaused
                  ? "rgba(34, 197, 94, 0.15)"
                  : "rgba(239, 68, 68, 0.15)",
                borderColor: isPaused
                  ? "rgba(34, 197, 94, 0.3)"
                  : "rgba(239, 68, 68, 0.3)",
                color: isPaused ? "#22c55e" : "#ef4444",
                border: isPaused
                  ? "1px solid rgba(34, 197, 94, 0.3)"
                  : "1px solid rgba(239, 68, 68, 0.3)",
                cursor: "pointer",
                padding: "6px 12px",
                borderRadius: "5px",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                fontSize: "11px",
                fontWeight: 600,
                transition: "all 0.2s",
              }}
            >
              {isPaused ? (
                <>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    stroke="currentColor"
                    strokeWidth="1"
                  >
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  Play
                </>
              ) : (
                <>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    stroke="currentColor"
                    strokeWidth="1"
                  >
                    <rect x="6" y="4" width="4" height="16" />
                    <rect x="14" y="4" width="4" height="16" />
                  </svg>
                  Pause
                </>
              )}
            </button>
            <button
              className={styles.refreshBtn}
              onClick={forceRefresh}
              title="Refresh now"
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="23 4 23 10 17 10" />
                <polyline points="1 20 1 14 7 14" />
                <path d="M3.5 9a9 9 0 0 1 14.8-3.3L23 10M1 14l4.7 4.3A9 9 0 0 0 20.5 15" />
              </svg>
            </button>
            <button
              className={styles.addBtn}
              onClick={() => setShowAddModal(true)}
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              Add Camera
            </button>
          </div>
        </div>

        {/* Alert banner */}
        <AlertBanner alerts={alerts} />

        {/* Stats */}
        <div className={styles.statsRow}>
          <div style={{ position: "relative" }}>
            <StatCard
              label="Total People"
              value={`${totalPeople}/${capacity}`}
              color={alertLevel?.color || "#00e5cc"}
              icon={
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <circle cx="9" cy="7" r="3" />
                  <path d="M3 21v-2a6 6 0 0 1 6-6" />
                  <circle cx="17" cy="7" r="3" />
                  <path d="M21 21v-2a6 6 0 0 0-6-6" />
                </svg>
              }
            />
            <button
              onClick={handleResetGlobalCount}
              disabled={resetLoading}
              title="Reset global count to 0"
              style={{
                position: "absolute",
                top: "8px",
                right: "8px",
                background: "rgba(100, 116, 139, 0.2)",
                border: "1px solid rgba(100, 116, 139, 0.4)",
                color: "#94a3b8",
                cursor: resetLoading ? "not-allowed" : "pointer",
                padding: "6px 10px",
                borderRadius: "5px",
                fontSize: "11px",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: "4px",
                transition: "all 0.2s",
                opacity: resetLoading ? 0.6 : 1,
              }}
              onMouseOver={(e) => {
                if (!resetLoading) {
                  e.target.style.background = "rgba(0, 229, 204, 0.1)";
                  e.target.style.color = "#00e5cc";
                  e.target.style.borderColor = "rgba(0, 229, 204, 0.4)";
                }
              }}
              onMouseOut={(e) => {
                if (!resetLoading) {
                  e.target.style.background = "rgba(100, 116, 139, 0.2)";
                  e.target.style.color = "#94a3b8";
                  e.target.style.borderColor = "rgba(100, 116, 139, 0.4)";
                }
              }}
            >
              {resetLoading ? (
                <>
                  <div
                    style={{
                      width: "10px",
                      height: "10px",
                      border: "2px solid rgba(255, 255, 255, 0.08)",
                      borderTopColor: "#00e5cc",
                      borderRadius: "50%",
                      animation: "spin 1s linear infinite",
                    }}
                  />
                  <span>Resetting...</span>
                </>
              ) : (
                "Reset"
              )}
            </button>
          </div>
          <div className={styles.statCard}>
            <div
              className={styles.statIcon}
              style={{ color: alertLevel?.color || "#00e5cc" }}
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <line x1="9" y1="9" x2="15" y2="9" />
                <line x1="9" y1="15" x2="15" y2="15" />
              </svg>
            </div>
            <div>
              <p
                className={styles.statValue}
                style={{ color: alertLevel?.color || "#00e5cc" }}
              >
                {`${occupancyPercent}%`}
              </p>
              <p className={styles.statLabel}>Area Capacity</p>
            </div>
          </div>
          <StatCard
            label="Active Cameras"
            value={cameras.length}
            color="#7dd3fc"
            icon={
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <path d="M23 7l-7 5 7 5V7z" />
                <rect x="1" y="5" width="15" height="14" rx="2" />
              </svg>
            }
          />
          <StatCard
            label="Alert Status"
            value={alertLevel?.label || "Safe"}
            color={alertLevel?.color || "#22c55e"}
            icon={
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <path d="M10.3 21h3.4M12 3a7 7 0 0 1 7 7c0 2.5 1 4 1 6H4c0-2 1-3.5 1-6a7 7 0 0 1 7-7z" />
              </svg>
            }
          />
        </div>

        {/* Error */}
        {fetchError && (
          <div className={styles.errorBox}>
            <strong>⚠️ Error:</strong> {fetchError}
            <br />
            <small>Check browser console (F12) for more details</small>
          </div>
        )}

        {/* Camera grid */}
        {loadingCams ? (
          <div className={styles.loadingState}>
            <div className={styles.spinner} />
            Loading cameras…
          </div>
        ) : cameras.length === 0 ? (
          <div className={styles.emptyState}>
            <svg
              width="52"
              height="52"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#1e2a3a"
              strokeWidth="0.8"
            >
              <path d="M23 7l-7 5 7 5V7z" />
              <rect x="1" y="5" width="15" height="14" rx="2" />
            </svg>
            <h3 className={styles.emptyTitle}>No cameras yet</h3>
            <p className={styles.emptyText}>
              Register your first camera to start monitoring crowd density.
            </p>
            <button
              className={styles.addBtn}
              onClick={() => setShowAddModal(true)}
            >
              Add First Camera
            </button>
          </div>
        ) : (
          <div className={styles.grid}>
            {cameras.map((cam) => (
              <CameraCard
                key={cam.id}
                camera={cam}
                countData={counts[cam.id]}
                onDelete={handleDelete}
                isPaused={isPaused}
              />
            ))}
          </div>
        )}

        {/* Analytics Section - Charts below cameras */}
        {!isPaused && cameras.length > 0 && (
          <div style={{ marginTop: "40px" }}>
            {/* Predictions */}
            <PredictionCard data={globalHistory} capacity={capacity} />

            {/* Global Trend Chart */}
            <GlobalTrendChart data={globalHistory} capacity={capacity} />

            {/* Camera Trend Charts */}
            {cameraHistory && Object.keys(cameraHistory).length > 0 && (
              <div
                style={{
                  marginTop: "32px",
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))",
                  gap: "24px",
                }}
              >
                {cameras.map(
                  (cam) =>
                    cameraHistory[cam.id] &&
                    cameraHistory[cam.id].length > 0 && (
                      <CameraTrendChart
                        key={`trend-${cam.id}`}
                        cameraId={cam.id}
                        cameraName={cam.name}
                        data={cameraHistory[cam.id]}
                        capacity={capacity}
                      />
                    ),
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {showAddModal && (
        <AddCameraModal
          onClose={() => setShowAddModal(false)}
          onSuccess={handleCameraAdded}
        />
      )}
    </div>
  );
};

export default DashboardPage;
