import { useEffect, useRef, useState } from "react";
import {
  detectHands,
  disposeHandLandmarker,
  getHandLandmarker,
} from "../services/handLandmarks";

/**
 * Run MediaPipe HandLandmarker on a live video element.
 * Frame results stay in a ref so the overlay can draw without a React render
 * on every video frame. HUD state is throttled.
 */
export function useHandLandmarks(videoRef, enabled) {
  const [hands, setHands] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);
  const handsRef = useRef([]);
  const landmarkerRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      handsRef.current = [];
      setHands([]);
      setFps(0);
      setStatus((current) => (current === "error" ? current : "idle"));
      return undefined;
    }

    let cancelled = false;
    let rafId = 0;
    let lastTimestamp = -1;
    const frameTimes = [];
    let lastHudUpdate = 0;

    setStatus("loading");
    setError(null);

    getHandLandmarker()
      .then((instance) => {
        if (cancelled) return;
        landmarkerRef.current = instance;
        setStatus("ready");

        const tick = () => {
          if (cancelled) return;
          const video = videoRef.current;
          if (video && video.readyState >= 2) {
            const now = performance.now();
            if (now !== lastTimestamp) {
              lastTimestamp = now;
              const detected = detectHands(instance, video, now);
              handsRef.current = detected;
              frameTimes.push(now);
              while (frameTimes.length > 0 && now - frameTimes[0] > 1000) {
                frameTimes.shift();
              }
              if (now - lastHudUpdate > 200) {
                lastHudUpdate = now;
                setHands(detected);
                setFps(frameTimes.length);
                setStatus("running");
              }
            }
          }
          rafId = window.requestAnimationFrame(tick);
        };

        rafId = window.requestAnimationFrame(tick);
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus("error");
        setError(
          err?.message ||
            "Could not load MediaPipe hand detection. Check the network and refresh."
        );
      });

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(rafId);
      handsRef.current = [];
    };
  }, [enabled, videoRef]);

  useEffect(() => () => disposeHandLandmarker(), []);

  return {
    hands,
    handsRef,
    status,
    error,
    fps,
  };
}
