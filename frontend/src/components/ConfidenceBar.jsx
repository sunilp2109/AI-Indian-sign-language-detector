export default function ConfidenceBar({ value = 0, threshold = 0.8, low = false }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const below = low || percent < Math.round(threshold * 100);

  return (
    <div className={`confidence ${below && percent > 0 ? "confidence-low" : ""}`}>
      <div className="confidence-header">
        <span>Confidence</span>
        <span className="confidence-percent">{percent}%</span>
      </div>
      <div
        className="confidence-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={`${percent} percent confidence${below && percent > 0 ? ", below threshold" : ""}`}
        aria-label="Prediction confidence"
      >
        <div
          className="confidence-fill"
          style={{ width: `${percent}%` }}
        />
        <span className="confidence-threshold" style={{ left: `${threshold * 100}%` }} />
      </div>
      <p className="confidence-caption">
        Accept signs at {Math.round(threshold * 100)}% or higher
        {below && percent > 0 ? " · below threshold" : ""}
      </p>
    </div>
  );
}
