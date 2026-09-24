import { useEffect, useRef } from "react";
import { drawHands } from "../utils/drawLandmarks";

export default function LandmarkOverlay({ handsRef, videoRef }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef?.current;
    if (!canvas || !video) {
      return undefined;
    }

    let rafId = 0;
    const tick = () => {
      const width = video.videoWidth || video.clientWidth;
      const height = video.videoHeight || video.clientHeight;
      if (width && height) {
        if (canvas.width !== width) canvas.width = width;
        if (canvas.height !== height) canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          drawHands(ctx, handsRef?.current || [], width, height);
        }
      }
      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [handsRef, videoRef]);

  return <canvas ref={canvasRef} className="landmark-overlay" aria-hidden="true" />;
}
