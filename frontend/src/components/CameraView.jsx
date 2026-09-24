import LandmarkOverlay from "./LandmarkOverlay";
import { summarizeHands } from "../utils/handFormat";

function cameraCopy(status, error) {
  switch (status) {
    case "requesting":
      return {
        title: "Allow camera access",
        body: "The browser should prompt for permission. Video stays in this tab and is not recorded.",
      };
    case "denied":
      return {
        title: "Permission denied",
        body: error,
      };
    case "unavailable":
      return {
        title: "Camera unavailable",
        body: error,
      };
    case "error":
      return {
        title: "Camera error",
        body: error,
      };
    case "stopped":
      return {
        title: "Camera stopped",
        body: "Start the camera again to resume live landmarks. Nothing was saved.",
      };
    default:
      return {
        title: "Camera idle",
        body: "Start the camera to see a live preview and hand landmarks. Video is not recorded.",
      };
  }
}

export default function CameraView({
  videoRef,
  cameraStatus,
  cameraError,
  onStart,
  onStop,
  hands,
  handsRef,
  fps,
  detectorStatus,
  detectorError,
}) {
  const isLive = cameraStatus === "live";
  const copy = cameraCopy(cameraStatus, cameraError);
  const handText = summarizeHands(hands);
  const noHands = isLive && hands.length === 0 && detectorStatus === "running";
  const detectorText =
    detectorStatus === "loading"
      ? "Loading MediaPipe…"
      : detectorStatus === "error"
        ? detectorError
        : detectorStatus === "running"
          ? `${fps} FPS`
          : detectorStatus === "ready"
            ? "Detector ready"
            : "Detector idle";

  return (
    <div className="camera-block">
      <div className="camera-stage">
        <video
          ref={videoRef}
          className="camera-video"
          playsInline
          muted
          autoPlay
          aria-label="Live webcam preview"
        />
        {isLive ? <LandmarkOverlay handsRef={handsRef} videoRef={videoRef} /> : null}

        {!isLive ? (
          <div className="camera-placeholder camera-placeholder-overlay">
            <p className="camera-title">{copy.title}</p>
            <p>{copy.body}</p>
          </div>
        ) : null}

        {isLive ? (
          <div className="camera-hud" aria-live="polite">
            <span>{detectorText}</span>
            <span>{handText}</span>
          </div>
        ) : null}

        {noHands ? (
          <p className="no-hands-banner" role="status">
            No hands detected — show one or both hands to the camera.
          </p>
        ) : null}
      </div>

      <div className="controls camera-controls">
        <button
          type="button"
          onClick={onStart}
          disabled={isLive || cameraStatus === "requesting"}
        >
          {cameraStatus === "requesting" ? "Requesting…" : "Start camera"}
        </button>
        <button type="button" className="secondary" onClick={onStop} disabled={!isLive}>
          Stop camera
        </button>
      </div>

      {detectorStatus === "error" && detectorError ? (
        <p className="hint" role="alert">
          {detectorError}
        </p>
      ) : (
        <p className="hint">
          Live frames are processed in the browser for hand landmarks only. This
          app does not record or upload video.
        </p>
      )}
    </div>
  );
}
