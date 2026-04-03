import { useState } from "react";
import {
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import styles from "./TrendChart.module.css";
import { getNiceAxisConfig } from "./chartScale";

const GlobalDensityChart = ({ data = [] }) => {
  const [showFullHistory, setShowFullHistory] = useState(false);

  if (!data || data.length === 0) {
    return (
      <div className={styles.chartContainer}>
        <div className={styles.chartHeader}>
          <h3 className={styles.chartTitle}>Approximate Density Trend</h3>
        </div>
        <div className={styles.emptyState}>
          No density data yet. Enable density monitoring on at least one camera.
        </div>
      </div>
    );
  }

  const displayData = showFullHistory ? data : data.slice(-10);
  const axisConfig = getNiceAxisConfig(displayData.map((point) => point.value));

  return (
    <div className={styles.chartContainer}>
      <div className={styles.chartHeader}>
        <h3 className={styles.chartTitle}>Approximate Density Trend</h3>
        {data.length > 10 && (
          <button
            className={styles.viewToggle}
            onClick={() => setShowFullHistory(!showFullHistory)}
          >
            {showFullHistory ? "Show Last 10" : "View All"}
          </button>
        )}
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={displayData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="densityTrendFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ffb84d" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#ffb84d" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="time" stroke="#94a3b8" style={{ fontSize: "12px" }} />
          <YAxis
            stroke="#94a3b8"
            style={{ fontSize: "12px" }}
            domain={axisConfig.domain}
            ticks={axisConfig.ticks}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(15, 23, 42, 0.95)",
              border: "1px solid rgba(255, 184, 77, 0.3)",
              borderRadius: "8px",
              color: "#e2e8f0",
            }}
            formatter={(value) => [Number(value).toFixed(2), "Density Score"]}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="none"
            fill="url(#densityTrendFill)"
            animationDuration={650}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#ffb84d"
            strokeWidth={3}
            dot={{ fill: "#ffb84d", r: 4 }}
            activeDot={{ r: 7, fill: "#ffffff", stroke: "#ffb84d", strokeWidth: 2 }}
            animationDuration={650}
          />
          <ReferenceLine y={2} stroke="rgba(34,197,94,0.45)" strokeDasharray="5 5" />
          <ReferenceLine y={3.5} stroke="rgba(245,158,11,0.45)" strokeDasharray="5 5" />
          <ReferenceLine y={5} stroke="rgba(239,68,68,0.45)" strokeDasharray="5 5" />
        </LineChart>
      </ResponsiveContainer>

      <div className={styles.chartStats}>
        <div className={styles.statItem}>
          <span className={styles.label}>Peak Score:</span>
          <span className={styles.value}>
            {Math.max(...data.map((d) => d.value)).toFixed(2)}
          </span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.label}>Current:</span>
          <span className={styles.value}>{data[data.length - 1].value.toFixed(2)}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.label}>Average:</span>
          <span className={styles.value}>
            {(data.reduce((sum, d) => sum + d.value, 0) / data.length).toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default GlobalDensityChart;
