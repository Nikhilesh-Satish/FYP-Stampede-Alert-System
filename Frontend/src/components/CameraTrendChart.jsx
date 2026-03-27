import {
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from "recharts";
import styles from "./TrendChart.module.css";

const CameraTrendChart = ({ cameraId, cameraName, data = [], capacity }) => {
  if (!data || data.length === 0) {
    return (
      <div className={styles.chartContainer}>
        <h3 className={styles.chartTitle}>{cameraName} - Trend</h3>
        <div className={styles.emptyState}>
          No data yet. Start monitoring to see trends.
        </div>
      </div>
    );
  }

  // Show last 10 data points for individual camera
  const displayData = data.slice(-10);

  return (
    <div className={styles.chartContainer}>
      <h3 className={styles.chartTitle}>{cameraName} - Trend</h3>

      <ResponsiveContainer width="100%" height={250}>
        <LineChart
          data={displayData}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <defs>
            <linearGradient id={`cameraTrendFill-${cameraId}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2fd8ff" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#2fd8ff" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          {capacity ? (
            <ReferenceArea
              y1={capacity * 0.8}
              y2={capacity}
              fill="rgba(255,184,77,0.08)"
            />
          ) : null}
          <XAxis dataKey="time" stroke="#94a3b8" style={{ fontSize: "11px" }} />
          <YAxis
            stroke="#94a3b8"
            style={{ fontSize: "11px" }}
            domain={[0, capacity]}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(15, 23, 42, 0.95)",
              border: "1px solid rgba(0, 229, 204, 0.3)",
              borderRadius: "8px",
              color: "#e2e8f0",
            }}
            formatter={(value) => [value, "People Count"]}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke="none"
            fill={`url(#cameraTrendFill-${cameraId})`}
            animationDuration={650}
          />
          <Line
            type="monotone"
            dataKey="count"
            stroke="#00e5cc"
            strokeWidth={2.5}
            dot={{ fill: "#00e5cc", r: 3 }}
            activeDot={{ r: 6, fill: "#ffffff", stroke: "#00e5cc", strokeWidth: 2 }}
            animationDuration={650}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className={styles.chartStats}>
        <div className={styles.statItem}>
          <span className={styles.label}>Peak:</span>
          <span className={styles.value}>
            {Math.max(...data.map((d) => d.count))}
          </span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.label}>Current:</span>
          <span className={styles.value}>{data[data.length - 1].count}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.label}>Avg:</span>
          <span className={styles.value}>
            {Math.round(
              data.reduce((sum, d) => sum + d.count, 0) / data.length,
            )}
          </span>
        </div>
      </div>
    </div>
  );
};

export default CameraTrendChart;
