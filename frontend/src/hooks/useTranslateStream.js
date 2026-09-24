import { useEffect, useRef, useState } from "react";
import {
  connectTranslatorSocket,
  sendTranslatorMessage,
  serializeHands,
} from "../services/websocket";

/**
 * Stream landmark frames to /ws/translate and keep the latest prediction.
 * Reconnects if the socket drops while the camera is still live.
 */
export function useTranslateStream({ enabled, handsRef, intervalMs = 50 }) {
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Stream idle");
  const socketRef = useRef(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  useEffect(() => {
    if (!enabled) {
      setStatus("idle");
      setMessage("Stream idle");
      setPrediction(null);
      return undefined;
    }

    let cancelled = false;
    let retryTimer = 0;
    let sendTimer = 0;
    let delay = 500;

    function clearSendTimer() {
      if (sendTimer) {
        window.clearInterval(sendTimer);
        sendTimer = 0;
      }
    }

    function connect() {
      if (cancelled) return;
      setStatus("checking");
      setMessage("Connecting WebSocket…");

      const socket = connectTranslatorSocket({
        onOpen() {
          if (cancelled) return;
          delay = 500;
          setStatus("live");
          setMessage("Streaming landmarks");
        },
        onMessage(event) {
          if (cancelled) return;
          try {
            const data = JSON.parse(event.data);
            setPrediction(data);
            if (data?.type === "hello" && data.ready === false) {
              setMessage(data.message || "Model is not loaded");
            }
          } catch {
            // Ignore a malformed server frame; keep the last good prediction.
          }
        },
        onError() {
          if (cancelled) return;
          setStatus("disconnected");
          setMessage("WebSocket error");
        },
        onClose() {
          socketRef.current = null;
          clearSendTimer();
          if (cancelled || !enabledRef.current) return;
          setStatus("disconnected");
          setMessage("WebSocket disconnected — retrying");
          retryTimer = window.setTimeout(connect, delay);
          delay = Math.min(delay * 2, 5000);
        },
      });

      socketRef.current = socket;
      clearSendTimer();
      sendTimer = window.setInterval(() => {
        const current = socketRef.current;
        const hands = Array.isArray(handsRef?.current) ? handsRef.current : [];
        sendTranslatorMessage(current, {
          type: "landmarks",
          hands: serializeHands(hands),
        });
      }, intervalMs);
    }

    connect();

    return () => {
      cancelled = true;
      window.clearTimeout(retryTimer);
      clearSendTimer();
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        sendTranslatorMessage(socket, { type: "reset" });
        try {
          socket.close();
        } catch {
          // Already closed.
        }
      }
    };
  }, [enabled, handsRef, intervalMs]);

  function send(payload) {
    return sendTranslatorMessage(socketRef.current, payload);
  }

  return { prediction, status, message, send };
}
