import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { HAND_LANDMARKER_MODEL_URL, MEDIAPIPE_WASM_URL } from "../config";
import {
  EXPECTED_LANDMARK_COUNT,
  formatHandResults,
  HAND_CONNECTIONS,
  summarizeHands,
} from "../utils/handFormat";

export { EXPECTED_LANDMARK_COUNT, formatHandResults, HAND_CONNECTIONS, summarizeHands };

let landmarker = null;
let loadingPromise = null;

async function createLandmarker(delegate) {
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL.replace(/\/$/, ""));
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: HAND_LANDMARKER_MODEL_URL,
      delegate,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

export async function getHandLandmarker() {
  if (landmarker) {
    return landmarker;
  }
  if (loadingPromise) {
    return loadingPromise;
  }

  loadingPromise = (async () => {
    try {
      landmarker = await createLandmarker("GPU");
    } catch {
      landmarker = await createLandmarker("CPU");
    }
    return landmarker;
  })().catch((error) => {
    loadingPromise = null;
    throw error;
  });

  return loadingPromise;
}

export function detectHands(instance, video, timestampMs) {
  if (!instance || !video || video.readyState < 2 || video.videoWidth === 0) {
    return [];
  }

  try {
    const result = instance.detectForVideo(video, timestampMs);
    return formatHandResults(result).filter(
      (hand) => hand.landmarks.length === EXPECTED_LANDMARK_COUNT
    );
  } catch {
    return [];
  }
}

export function disposeHandLandmarker() {
  try {
    landmarker?.close?.();
  } catch {
    // MediaPipe close is best-effort on hot reload.
  }
  landmarker = null;
  loadingPromise = null;
}
