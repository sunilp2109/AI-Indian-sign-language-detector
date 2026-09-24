function Badge({ label, status, message }) {
  return (
    <li className={`status-badge status-${status}`} title={message}>
      <span className="status-dot" aria-hidden="true" />
      <span>
        <strong>{label}</strong>
        <span className="status-text">{message}</span>
      </span>
    </li>
  );
}

export default function StatusIndicator({
  cameraStatus,
  cameraMessage,
  connectionStatus,
  connectionMessage,
  modelStatus,
  modelMessage,
}) {
  return (
    <ul className="status-row" aria-label="System status">
      <Badge label="Camera" status={cameraStatus} message={cameraMessage} />
      <Badge label="Connection" status={connectionStatus} message={connectionMessage} />
      <Badge label="Model" status={modelStatus} message={modelMessage} />
    </ul>
  );
}
