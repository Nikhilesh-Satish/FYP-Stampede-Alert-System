import { useMemo } from "react";
import styles from "./PredictionCard.module.css";

const DEFAULT_SAMPLE_INTERVAL_MS = 10 * 1000;
const TEN_MINUTES_MS = 10 * 60 * 1000;
const FIFTY_MINUTES_MS = 50 * 60 * 1000;
const MIN_FORECAST_POINTS = 6;
const MIN_HISTORY_SPAN_MS = 60 * 1000;

const median = (arr) => {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
};

const PredictionCard = ({ data = [], capacity }) => {
  const prediction = useMemo(() => {
    if (data.length < 2) {
      return null;
    }

    const recentData = data.slice(-20);
    const values = recentData.map((d) => d.count);
    const timestamps = recentData.map((d) => d.timestamp);

    // Estimate sample interval from actual timestamps for true time-based forecasting.
    const intervals = [];
    for (let i = 1; i < timestamps.length; i += 1) {
      const prev = timestamps[i - 1];
      const curr = timestamps[i];
      if (Number.isFinite(prev) && Number.isFinite(curr) && curr > prev) {
        intervals.push(curr - prev);
      }
    }
    const avgIntervalMs =
      intervals.length > 0
        ? intervals.reduce((a, b) => a + b, 0) / intervals.length
        : DEFAULT_SAMPLE_INTERVAL_MS;

    const lastValue = values[values.length - 1];
    const historySpanMs = Math.max(0, timestamps[timestamps.length - 1] - timestamps[0]);

    // Stabilize early forecasting: until enough history is collected, keep estimate near current value.
    if (values.length < MIN_FORECAST_POINTS || historySpanMs < MIN_HISTORY_SPAN_MS) {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance =
        values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length;
      const stdDev = Math.sqrt(variance);
      return {
        trend: "0.00",
        trendPercent: "0.0",
        volatility: stdDev.toFixed(1),
        predictedNext: Math.max(0, Math.round(lastValue)),
        predictedNext5: Math.max(0, Math.round(lastValue)),
        riskLevel: lastValue > capacity ? "Critical" : "Low",
        riskColor: lastValue > capacity ? "#dc2626" : "#10b981",
        riskMessage:
          lastValue > capacity
            ? `⚠️ CAPACITY EXCEEDED: Currently over safe limit by ${Math.round(lastValue - capacity)} people`
            : "Collecting more history for stable forecasting",
        avgCurrent: mean.toFixed(1),
      };
    }

    // Robust slope (Theil-Sen): median of pairwise slopes in people/ms.
    const pairwiseSlopes = [];
    for (let i = 0; i < values.length - 1; i += 1) {
      for (let j = i + 1; j < values.length; j += 1) {
        const dt = timestamps[j] - timestamps[i];
        if (dt > 0) {
          pairwiseSlopes.push((values[j] - values[i]) / dt);
        }
      }
    }
    const slopePerMs = median(pairwiseSlopes);

    // Also compute recent absolute rate to clamp impossible forecast jumps from transient glitches.
    const absRates = [];
    for (let i = 1; i < values.length; i += 1) {
      const dt = timestamps[i] - timestamps[i - 1];
      if (dt > 0) {
        absRates.push(Math.abs((values[i] - values[i - 1]) / dt));
      }
    }
    const medianAbsRatePerMs = median(absRates);

    const maxDelta10 = Math.max(100, medianAbsRatePerMs * TEN_MINUTES_MS * 3);
    const maxDelta50 = Math.max(200, medianAbsRatePerMs * FIFTY_MINUTES_MS * 3);

    const clamp = (val, min, max) => Math.min(max, Math.max(min, val));

    const predictedNextRaw = lastValue + slopePerMs * TEN_MINUTES_MS;
    const predictedNext5Raw = lastValue + slopePerMs * FIFTY_MINUTES_MS;
    const predictedNext = Math.max(
      0,
      Math.round(clamp(predictedNextRaw, lastValue - maxDelta10, lastValue + maxDelta10)),
    );
    const predictedNext5 = Math.max(
      0,
      Math.round(clamp(predictedNext5Raw, lastValue - maxDelta50, lastValue + maxDelta50)),
    );

    // Keep displayed trend in people per sample step for UI continuity.
    const trendPerStep = slopePerMs * avgIntervalMs;

    // Legacy stats used elsewhere in the card.
    const n = values.length;
    const yMean = values.reduce((a, b) => a + b, 0) / n;
    const trendPercent = ((trendPerStep / (yMean + 1)) * 100).toFixed(1);

    // Calculate volatility (standard deviation)
    const mean = yMean;
    const variance =
      values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

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

    const projected10MinDelta = predictedNext - lastValue;
    if (Math.abs(projected10MinDelta) > mean * 0.1 && lastValue <= capacity) {
      riskMessage =
        projected10MinDelta > 0
          ? "⬆️ Crowd size is increasing"
          : "⬇️ Crowd size is decreasing";
    }

    return {
      trend: trendPerStep.toFixed(2),
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
