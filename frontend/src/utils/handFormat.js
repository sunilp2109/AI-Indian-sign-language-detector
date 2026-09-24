/** MediaPipe 21-point hand skeleton (wrist, thumb, index, middle, ring, pinky). */
export const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

export const EXPECTED_LANDMARK_COUNT = 21;

export function normalizeHandLabel(name) {
  const value = String(name || "").toLowerCase();
  if (value === "left") return "Left";
  if (value === "right") return "Right";
  return "Unknown";
}

/**
 * Convert a MediaPipe HandLandmarkerResult into a stable, UI-friendly list.
 * Never throws on missing hands or incomplete landmark arrays.
 */
export function formatHandResults(result) {
  if (!result || !Array.isArray(result.landmarks)) {
    return [];
  }

  return result.landmarks.map((points, index) => {
    const category =
      result.handedness?.[index]?.[0] ||
      result.handednesses?.[index]?.[0] ||
      {};

    const landmarks = Array.isArray(points)
      ? points.map((point) => ({
          x: Number.isFinite(point?.x) ? point.x : 0,
          y: Number.isFinite(point?.y) ? point.y : 0,
          z: Number.isFinite(point?.z) ? point.z : 0,
        }))
      : [];

    return {
      label: normalizeHandLabel(category.categoryName),
      score: Number.isFinite(category.score) ? category.score : 0,
      landmarks,
    };
  });
}

export function summarizeHands(hands) {
  if (!hands || hands.length === 0) {
    return "No hands detected";
  }

  const labels = hands.map((hand) => hand.label);
  const unique = [...new Set(labels)];
  if (unique.length === 1 && unique[0] !== "Unknown") {
    return hands.length === 1
      ? `${unique[0]} hand`
      : `${hands.length} ${unique[0].toLowerCase()} hands`;
  }
  if (labels.includes("Left") && labels.includes("Right")) {
    return "Left + Right hands";
  }
  return `${hands.length} hand${hands.length === 1 ? "" : "s"}`;
}
