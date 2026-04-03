import { useState } from "react";
import { cameraApi } from "../api/services";
import styles from "./AddCameraModal.module.css";

const AddCameraModal = ({ onClose, onSuccess }) => {
  const [form, setForm] = useState({
    name: "",
    streamPath: "",
    shotType: "ground",
    countAxis: "x",
    inDirection: "positive",
    processingPreset: "balanced",
    countingEnabled: true,
    densityEnabled: false,
    monitoredAreaSqm: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    setError("");
  };

  const directionOptions =
    form.countAxis === "y"
      ? [
          { value: "positive", label: "Top to bottom means IN" },
          { value: "negative", label: "Bottom to top means IN" },
        ]
      : [
          { value: "positive", label: "Left to right means IN" },
          { value: "negative", label: "Right to left means IN" },
        ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.name.trim() || !form.streamPath.trim()) {
      setError("All fields are required.");
      return;
    }
    if (form.densityEnabled) {
      const area = Number(form.monitoredAreaSqm);
      if (!Number.isFinite(area) || area <= 0) {
        setError("Enter a valid monitored area in square meters for density monitoring.");
        return;
      }
    }
    setLoading(true);
    try {
      console.log("Adding camera:", form);
      const result = await cameraApi.addCamera(form);
      console.log("Add camera response:", result);
      onClose();
      setTimeout(() => {
        onSuccess(result);
      }, 0);
    } catch (err) {
      console.error("Add camera error:", err);
      setError(err.message || "Failed to add camera.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <div className={styles.iconWrap}>
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#00e5cc"
              strokeWidth="1.5"
            >
              <path d="M23 7l-7 5 7 5V7z" />
              <rect x="1" y="5" width="15" height="14" rx="2" />
            </svg>
          </div>
          <div>
            <h2 className={styles.title}>Register Camera</h2>
            <p className={styles.subtitle}>Add a new surveillance feed</p>
          </div>
          <button className={styles.closeBtn} onClick={onClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label className={styles.label}>Camera Name</label>
            <input
              className={styles.input}
              type="text"
              name="name"
              value={form.name}
              onChange={handleChange}
              placeholder="e.g. Gate A — Entry Camera"
              autoFocus
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Video Stream Path</label>
            <input
              className={styles.input}
              type="text"
              name="streamPath"
              value={form.streamPath}
              onChange={handleChange}
              placeholder="/streams/gate_a.mp4  or  rtsp://…"
            />
            <p className={styles.hint}>
              Local file path or RTSP/HTTP stream URL processed by your backend.
            </p>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Shot Type</label>
            <select
              className={styles.input}
              name="shotType"
              value={form.shotType}
              onChange={handleChange}
            >
              <option value="ground">Ground / normal camera</option>
              <option value="drone">Drone / overhead camera</option>
            </select>
            <p className={styles.hint}>
              Drone mode uses overhead-style detection/counting. Ground mode uses normal perspective tracking.
            </p>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Counting Direction</label>
            <select
              className={styles.input}
              name="countAxis"
              value={form.countAxis}
              onChange={handleChange}
              disabled={!form.countingEnabled}
            >
              <option value="x">Left to right / right to left</option>
              <option value="y">Top to bottom / bottom to top</option>
            </select>
            <p className={styles.hint}>
              Use top-to-bottom for overhead or drone-style feeds. Use left-to-right for side-view feeds.
            </p>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>IN Direction</label>
            <select
              className={styles.input}
              name="inDirection"
              value={form.inDirection}
              onChange={handleChange}
              disabled={!form.countingEnabled}
            >
              {directionOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <p className={styles.hint}>
              The opposite movement will automatically be counted as OUT.
            </p>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Entry / Exit Counting</label>
            <select
              className={styles.input}
              name="countingEnabled"
              value={String(form.countingEnabled)}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  countingEnabled: e.target.value === "true",
                }))
              }
            >
              <option value="true">Enabled</option>
              <option value="false">Disabled</option>
            </select>
            <p className={styles.hint}>
              Disable this if the camera should only monitor crowd density and not line crossings.
            </p>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Crowd Density Monitoring</label>
            <select
              className={styles.input}
              name="densityEnabled"
              value={String(form.densityEnabled)}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  densityEnabled: e.target.value === "true",
                }))
              }
            >
              <option value="false">Disabled</option>
              <option value="true">Enabled</option>
            </select>
            <p className={styles.hint}>
              Uses perspective-adjusted weighting to estimate crowd density and raise stampede-risk alerts.
            </p>
          </div>

          {form.densityEnabled && (
            <div className={styles.field}>
              <label className={styles.label}>Monitored Area (m²)</label>
              <input
                className={styles.input}
                type="number"
                min="0.1"
                step="0.1"
                name="monitoredAreaSqm"
                value={form.monitoredAreaSqm}
                onChange={handleChange}
                placeholder="e.g. 45"
              />
              <p className={styles.hint}>
                Enter the actual crowd area covered by this camera, not the full visible scene.
              </p>
            </div>
          )}

          <div className={styles.field}>
            <label className={styles.label}>Processing Preset</label>
            <select
              className={styles.input}
              name="processingPreset"
              value={form.processingPreset}
              onChange={handleChange}
            >
              <option value="fast">Fast - lower latency, lower accuracy</option>
              <option value="balanced">Balanced - default tradeoff</option>
              <option value="accurate">Accurate - slower, stronger detection</option>
            </select>
            <p className={styles.hint}>
              Choose how aggressively the backend should trade speed for detection quality on this camera.
            </p>
          </div>

          {error && <p className={styles.error}>{error}</p>}

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.cancelBtn}
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              type="submit"
              className={styles.submitBtn}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className={styles.btnSpinner} />
                  Registering…
                </>
              ) : (
                "Add Camera"
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddCameraModal;
