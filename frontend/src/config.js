const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const WEBSOCKET_URL =
  import.meta.env.VITE_WEBSOCKET_URL || "ws://127.0.0.1:8000/ws/translate";

const MEDIAPIPE_WASM_URL =
  import.meta.env.VITE_MEDIAPIPE_WASM_URL || `${import.meta.env.BASE_URL}mediapipe/wasm`;

const HAND_LANDMARKER_MODEL_URL =
  import.meta.env.VITE_HAND_LANDMARKER_MODEL_URL ||
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export { API_URL, WEBSOCKET_URL, MEDIAPIPE_WASM_URL, HAND_LANDMARKER_MODEL_URL };
