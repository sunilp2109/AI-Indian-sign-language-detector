import { WEBSOCKET_URL } from "../config";

/**
 * Compact MediaPipe-style hands for the translator WebSocket.
 * Drops scores and unknown fields so each frame stays small.
 */
export function serializeHands(hands) {
  if (!Array.isArray(hands)) {
    return [];
  }

  return hands.flatMap((hand) => {
    const landmarks = Array.isArray(hand?.landmarks)
      ? hand.landmarks.map((point) => ({
          x: Number.isFinite(point?.x) ? point.x : 0,
          y: Number.isFinite(point?.y) ? point.y : 0,
          z: Number.isFinite(point?.z) ? point.z : 0,
        }))
      : [];
    if (landmarks.length !== 21) {
      return [];
    }
    return [{ label: hand?.label || "Unknown", landmarks }];
  });
}

export function connectTranslatorSocket({ onOpen, onMessage, onError, onClose } = {}) {
  const socket = new WebSocket(WEBSOCKET_URL);

  socket.addEventListener("open", (event) => onOpen?.(event));
  socket.addEventListener("message", (event) => onMessage?.(event));
  socket.addEventListener("error", (event) => onError?.(event));
  socket.addEventListener("close", (event) => onClose?.(event));

  return socket;
}

export function sendTranslatorMessage(socket, payload) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return false;
  }
  try {
    socket.send(JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}
