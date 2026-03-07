import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import styles from "./TrendChart.module.css";

const GlobalTrendChart = ({ data = [], capacity }) => {
  const [showFullHistory, setShowFullHistory] = useState(false);

  if (!data || data.length === 0) {
    return (
      <div className={styles.chartContainer}>
        <h3 className={styles.chartTitle}>Global Crowd Trend</h3>
        <div className={styles.emptyState}>
          No data yet. Start monitoring to see trends.
        </div>
      </div>
    );
  }

  const displayData = showFullHistory ? data : data.slice(-10);

  return (
    <div className={styles.chartContainer}>
      <div className={styles.chartHeader}>
        <h3 className={styles.chartTitle}>Global Crowd Trend</h3>
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
        <LineChart
          data={displayData}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="time" stroke="#94a3b8" style={{ fontSize: "12px" }} />
          <YAxis
            stroke="#94a3b8"
            style={{ fontSize: "12px" }}
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
          <Line
            type="monotone"
            dataKey="count"
            stroke="#00e5cc"
            strokeWidth={2}
            dot={{ fill: "#00e5cc", r: 4 }}
            activeDot={{ r: 6 }}
            animationDuration={300}
          />
          {capacity && (
            <Line
              type="monotone"
              dataKey={() => capacity}
              stroke="rgba(239, 68, 68, 0.5)"
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              animationDuration={0}
              name="Capacity"
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      <div className={styles.chartStats}>
        <div className={styles.statItem}>
          <span className={styles.label}>Peak Count:</span>
          <span className={styles.value}>
            {Math.max(...data.map((d) => d.count))}
          </span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.label}>Current:</span>
          <span className={styles.value}>{data[data.length - 1].count}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.label}>Average:</span>
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

export default GlobalTrendChart;
