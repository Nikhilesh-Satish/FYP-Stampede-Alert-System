import { useMemo } from "react";
import styles from "./PredictionCard.module.css";

const PredictionCard = ({ data = [], capacity }) => {
  const prediction = useMemo(() => {
    if (data.length < 2) {
      return null;
    }

    const recentData = data.slice(-10);
    const values = recentData.map((d) => d.count);

    // Calculate trend
    const firstHalf = values.slice(0, Math.floor(values.length / 2));
    const secondHalf = values.slice(Math.floor(values.length / 2));

    const avgFirst =
      firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length || 0;
    const avgSecond =
      secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length || 0;

    const trend = avgSecond - avgFirst;
    const trendPercent = ((trend / (avgFirst + 1)) * 100).toFixed(1);

    // Calculate volatility (standard deviation)
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance =
      values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    // Predict next value
    const lastValue = values[values.length - 1];
    const predictedNext = Math.max(0, Math.round(lastValue + trend));
    const predictedNext5 = Math.max(0, Math.round(lastValue + trend * 5));

    // Determine risk level
    let riskLevel = "Low";
    let riskColor = "#10b981"; // green
    let riskMessage = "Crowd levels are stable";

    // Check if already exceeded capacity
    if (lastValue > capacity) {
      riskLevel = "Critical";
      riskColor = "#dc2626"; // darker red
      riskMessage = `⚠️ CAPACITY EXCEEDED: Currently over safe limit by ${Math.round(lastValue - capacity)} people`;
    } else if (predictedNext5 > capacity) {
      riskLevel = "Critical";
      riskColor = "#ef4444"; // red
      riskMessage = `🚨 Will exceed capacity in 50 minutes`;
    } else if (predictedNext5 > capacity * 0.8) {
      riskLevel = "High";
      riskColor = "#ef4444"; // red
      riskMessage = "High risk: Approaching safe capacity limit";
    } else if (predictedNext5 > capacity * 0.6) {
      riskLevel = "Medium";
      riskColor = "#f59e0b"; // amber
      riskMessage = "Moderate rise in crowd levels detected";
    }

    if (trend > mean * 0.1 && lastValue <= capacity) {
      riskMessage =
        trend > 0
          ? "⬆️ Crowd size is increasing"
          : "⬇️ Crowd size is decreasing";
    }

    return {
      trend: trend.toFixed(1),
      trendPercent,
      volatility: stdDev.toFixed(1),
      predictedNext,
      predictedNext5,
      riskLevel,
      riskColor,
      riskMessage,
      avgCurrent: mean.toFixed(1),
    };
  }, [data, capacity]);

  if (data.length < 2) {
    return (
      <div className={styles.predictionCard}>
        <h3 className={styles.title}>Predictive Analytics</h3>
        <div className={styles.emptyState}>
          Need at least 2 data points to generate predictions. Keep monitoring.
        </div>
      </div>
    );
  }

  return (
    <div className={styles.predictionCard}>
      <h3 className={styles.title}>Predictive Analytics</h3>

      <div
        className={styles.riskIndicator}
        style={{ borderColor: prediction.riskColor }}
      >
        <div
          className={styles.riskLevel}
          style={{ color: prediction.riskColor }}
        >
          {prediction.riskLevel} Risk
        </div>
        <div className={styles.riskMessage}>{prediction.riskMessage}</div>
      </div>

      <div className={styles.metricsGrid}>
        <div className={styles.metricBox}>
          <div className={styles.metricLabel}>Current Average</div>
          <div className={styles.metricValue}>{prediction.avgCurrent}</div>
          <div className={styles.metricCapacity}>of {capacity} capacity</div>
        </div>

        <div className={styles.metricBox}>
          <div className={styles.metricLabel}>Trend (Recent)</div>
          <div
            className={styles.metricValue}
            style={{
              color: parseFloat(prediction.trend) > 0 ? "#ef4444" : "#10b981",
            }}
          >
            {prediction.trend > 0 ? "+" : ""}
            {prediction.trend}
          </div>
        </div>

        <div className={styles.metricBox}>
          <div className={styles.metricLabel}>Next 10 Min Estimate</div>
          <div className={styles.metricValue}>{prediction.predictedNext}</div>
        </div>

        <div className={styles.metricBox}>
          <div className={styles.metricLabel}>Peak Estimate (50 Min)</div>
          <div
            className={styles.metricValue}
            style={{
              color:
                prediction.predictedNext5 > capacity * 0.8
                  ? "#ef4444"
                  : prediction.predictedNext5 > capacity * 0.6
                    ? "#f59e0b"
                    : "#10b981",
            }}
          >
            {prediction.predictedNext5}
          </div>
          <div className={styles.metricSubtext}>
            {Math.round((prediction.predictedNext5 / capacity) * 100)}% of
            capacity
          </div>
        </div>
      </div>

      <div className={styles.disclaimer}>
        <span>💡</span> Predictions based on recent trend analysis. Actual
        values may vary.
      </div>
    </div>
  );
};

export default PredictionCard;
