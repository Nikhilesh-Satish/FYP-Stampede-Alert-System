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
import Reveal from "../components/Reveal";
import { usePointerGlow } from "../hooks/usePointerGlow";
import { useMagneticHover } from "../hooks/useMagneticHover";

const StatCard = ({ label, value, color, icon, children }) => (
  <div className={styles.statCard}>
    <div className={styles.statIcon} style={{ color }}>
      {icon}
    </div>
    <div className={styles.statBody}>
      <p className={styles.statValue} style={{ color }}>
        {value}
      </p>
      <p className={styles.statLabel}>{label}</p>
    </div>
    {children}
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

  useEffect(() => {
    cameraApi
      .getCameras()
      .then((data) => {
        setCameras(Array.isArray(data) ? data : data.cameras || []);
      })
      .catch((err) => {
        console.error("Failed to load cameras:", err);
        setFetchError("Could not load cameras. Is your backend running?");
      })
      .finally(() => setLoadingCams(false));
  }, []);

  const handleCameraAdded = (cam) => {
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

  const totalPeople = globalAlert?.totalCount ?? 0;
  const occupancyPercent = globalAlert?.occupancyPercent ?? 0;
  const alertLevel = globalAlert?.alertLevel;
  const heroGlow = usePointerGlow();
  const addMagnetic = useMagneticHover();

  return (
    <div className={styles.page}>
      <Navbar alertCount={alerts.length} />

      <main className={styles.main}>
        <section
          ref={heroGlow.ref}
          {...heroGlow.glowProps}
          className={`${styles.heroPanel} interactiveGlow`}
        >
          <div className={styles.heroParticles}>
            <span />
            <span />
            <span />
            <span />
          </div>
          <div className={styles.pageHeader}>
            <div>
              <p className={styles.kicker}>Operations console</p>
              <h1 className={styles.pageTitle}>Admin Control Panel</h1>
              <p className={styles.pageSubtitle}>
                Welcome back, <span className={styles.hi}>{user?.firstName || "Admin"}</span>
                {lastUpdated && (
                  <span className={styles.ts}>
                    Updated{" "}
                    {lastUpdated.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                )}
              </p>
            </div>

            <div className={styles.headerBtns}>
              <div className={styles.capacityCard}>
                <label className={styles.capacityLabel}>Area Capacity</label>
                <div className={styles.capacityInputWrap}>
                  <input
                    className={styles.capacityInput}
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
                      if (e.target.value === "" || parseInt(e.target.value, 10) < 1) {
                        setCapacity(DEFAULT_AREA_CAPACITY);
                      }
                    }}
                  />
                  <span className={styles.capacitySuffix}>people</span>
                </div>
              </div>

              <button
                className={`${styles.playPauseBtn} ${
                  isPaused ? styles.playState : styles.pauseState
                }`}
                onClick={() => setIsPaused(!isPaused)}
                title={
                  isPaused
                    ? "Start monitoring all cameras"
                    : "Pause monitoring all cameras"
                }
              >
                {isPaused ? (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    Play
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="4" width="4" height="16" />
                      <rect x="14" y="4" width="4" height="16" />
                    </svg>
                    Pause
                  </>
                )}
              </button>

              <button className={styles.refreshBtn} onClick={forceRefresh} title="Refresh now">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="23 4 23 10 17 10" />
                  <polyline points="1 20 1 14 7 14" />
                  <path d="M3.5 9a9 9 0 0 1 14.8-3.3L23 10M1 14l4.7 4.3A9 9 0 0 0 20.5 15" />
                </svg>
              </button>

              <button
                ref={addMagnetic.ref}
                {...addMagnetic.magneticProps}
                className={`${styles.addBtn} magneticButton`}
                onClick={() => setShowAddModal(true)}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
                Add Camera
              </button>
            </div>
          </div>

          <div className={styles.commandStrip}>
            <div className={styles.commandCard}>
              <span className={styles.commandLabel}>System mode</span>
              <strong>{isPaused ? "Standby" : "Active Monitoring"}</strong>
            </div>
            <div className={styles.commandCard}>
              <span className={styles.commandLabel}>Live alerts</span>
              <strong>{alerts.length}</strong>
            </div>
            <div className={styles.commandCard}>
              <span className={styles.commandLabel}>Tracked cameras</span>
              <strong>{cameras.length}</strong>
            </div>
          </div>
        </section>

        <Reveal>
          <AlertBanner alerts={alerts} />
        </Reveal>

        <div className={styles.statsRow}>
          <Reveal delay={0}>
            <StatCard
              label="Total People"
              value={`${totalPeople}/${capacity}`}
              color={alertLevel?.color || "#65f2d7"}
              icon={
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="9" cy="7" r="3" />
                  <path d="M3 21v-2a6 6 0 0 1 6-6" />
                  <circle cx="17" cy="7" r="3" />
                  <path d="M21 21v-2a6 6 0 0 0-6-6" />
                </svg>
              }
            >
              <button
                onClick={handleResetGlobalCount}
                disabled={resetLoading}
                title="Reset global count to 0"
                className={styles.miniAction}
              >
                {resetLoading ? (
                  <>
                    <div className={styles.inlineSpinner} />
                    Resetting
                  </>
                ) : (
                  "Reset"
                )}
              </button>
            </StatCard>
          </Reveal>

          <Reveal delay={70}>
            <StatCard
              label="Area Capacity"
              value={`${occupancyPercent}%`}
              color={alertLevel?.color || "#65f2d7"}
              icon={
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <line x1="9" y1="9" x2="15" y2="9" />
                  <line x1="9" y1="15" x2="15" y2="15" />
                </svg>
              }
            />
          </Reveal>

          <Reveal delay={140}>
            <StatCard
              label="Active Cameras"
              value={cameras.length}
              color="#7dd3fc"
              icon={
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M23 7l-7 5 7 5V7z" />
                  <rect x="1" y="5" width="15" height="14" rx="2" />
                </svg>
              }
            />
          </Reveal>

          <Reveal delay={210}>
            <StatCard
              label="Alert Status"
              value={alertLevel?.label || "Safe"}
              color={alertLevel?.color || "#4ade80"}
              icon={
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M10.3 21h3.4M12 3a7 7 0 0 1 7 7c0 2.5 1 4 1 6H4c0-2 1-3.5 1-6a7 7 0 0 1 7-7z" />
                </svg>
              }
            />
          </Reveal>
        </div>

        {fetchError && (
          <div className={styles.errorBox}>
            <strong>System Warning:</strong> {fetchError}
          </div>
        )}

        {loadingCams ? (
          <div className={styles.loadingState}>
            <div className={styles.spinner} />
            Loading cameras…
          </div>
        ) : cameras.length === 0 ? (
          <div className={styles.emptyState}>
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="1">
              <path d="M23 7l-7 5 7 5V7z" />
              <rect x="1" y="5" width="15" height="14" rx="2" />
            </svg>
            <h3 className={styles.emptyTitle}>No cameras registered yet</h3>
            <p className={styles.emptyText}>
              Add a feed to begin crowd monitoring and alert analysis.
            </p>
            <button className={styles.addBtn} onClick={() => setShowAddModal(true)}>
              Add First Camera
            </button>
          </div>
        ) : (
          <div className={styles.grid}>
            {cameras.map((cam) => (
              <Reveal key={cam.id} delay={0}>
                <CameraCard
                  camera={cam}
                  countData={counts[cam.id]}
                  onDelete={handleDelete}
                  isPaused={isPaused}
                />
              </Reveal>
            ))}
          </div>
        )}

        {!isPaused && cameras.length > 0 && (
          <section className={styles.analyticsSection}>
            <Reveal>
              <PredictionCard data={globalHistory} capacity={capacity} />
            </Reveal>
            <Reveal delay={70}>
              <GlobalTrendChart data={globalHistory} capacity={capacity} />
            </Reveal>

            {cameraHistory && Object.keys(cameraHistory).length > 0 && (
              <div className={styles.chartGrid}>
                {cameras.map(
                  (cam) =>
                    cameraHistory[cam.id] &&
                    cameraHistory[cam.id].length > 0 && (
                      <Reveal key={`trend-wrap-${cam.id}`} delay={100}>
                        <CameraTrendChart
                          key={`trend-${cam.id}`}
                          cameraId={cam.id}
                          cameraName={cam.name}
                          data={cameraHistory[cam.id]}
                          capacity={capacity}
                        />
                      </Reveal>
                    ),
                )}
              </div>
            )}
          </section>
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
