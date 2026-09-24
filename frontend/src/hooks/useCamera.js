import { useCallback, useEffect, useRef, useState } from "react";

function mapGetUserMediaError(error) {
  const name = error?.name || "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return {
      status: "denied",
      message: "Camera permission denied. Allow camera access in the browser and try again.",
    };
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return {
      status: "unavailable",
      message: "No camera was found on this device.",
    };
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return {
      status: "error",
      message: "The camera is already in use by another application.",
    };
  }
  if (name === "SecurityError") {
    return {
      status: "error",
      message: "Camera access is blocked on this page. Use http://localhost or HTTPS.",
    };
  }
  return {
    status: "error",
    message: error?.message || "Could not start the camera.",
  };
}

/**
 * Browser webcam capture. Frames stay in memory on the MediaStream;
 * this hook never records or downloads video.
 */
export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  const stopTracks = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const stop = useCallback(() => {
    stopTracks();
    setStatus((current) => (current === "idle" ? "idle" : "stopped"));
    setError(null);
  }, [stopTracks]);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("unavailable");
      setError("Camera API is not available in this browser.");
      return;
    }

    stopTracks();
    setStatus("requesting");
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        stopTracks();
        setStatus("error");
        setError("Camera preview element is not ready.");
        return;
      }

      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;

      await new Promise((resolve, reject) => {
        const onLoaded = async () => {
          try {
            await video.play();
            resolve();
          } catch (playError) {
            reject(playError);
          }
        };
        if (video.readyState >= 1) {
          onLoaded();
        } else {
          video.onloadedmetadata = onLoaded;
        }
        video.onerror = () => reject(new Error("The camera stream could not be played."));
      });

      setStatus("live");
    } catch (err) {
      stopTracks();
      const mapped = mapGetUserMediaError(err);
      setStatus(mapped.status);
      setError(mapped.message);
    }
  }, [stopTracks]);

  useEffect(() => () => stopTracks(), [stopTracks]);

  return {
    videoRef,
    status,
    error,
    start,
    stop,
  };
}
