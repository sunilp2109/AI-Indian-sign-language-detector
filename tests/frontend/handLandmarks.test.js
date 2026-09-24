import { describe, expect, it } from "vitest";
import {
  EXPECTED_LANDMARK_COUNT,
  formatHandResults,
  HAND_CONNECTIONS,
  normalizeHandLabel,
  summarizeHands,
} from "../../frontend/src/utils/handFormat.js";
import { toPixel } from "../../frontend/src/utils/drawLandmarks.js";

function fakeHand(x = 0.5, y = 0.5) {
  return Array.from({ length: EXPECTED_LANDMARK_COUNT }, () => ({ x, y, z: 0 }));
}

describe("hand landmark formatting", () => {
  it("returns an empty list when MediaPipe finds no hands", () => {
    expect(formatHandResults(null)).toEqual([]);
    expect(formatHandResults({})).toEqual([]);
    expect(formatHandResults({ landmarks: [] })).toEqual([]);
  });

  it("maps left and right hands without throwing on extra fields", () => {
    const formatted = formatHandResults({
      landmarks: [fakeHand(0.2, 0.3), fakeHand(0.8, 0.4)],
      handedness: [
        [{ categoryName: "Left", score: 0.97 }],
        [{ categoryName: "Right", score: 0.91 }],
      ],
    });

    expect(formatted).toHaveLength(2);
    expect(formatted[0].label).toBe("Left");
    expect(formatted[1].label).toBe("Right");
    expect(formatted[0].landmarks).toHaveLength(21);
    expect(formatted[1].score).toBeCloseTo(0.91);
  });

  it("tolerates missing coordinates and unknown handedness", () => {
    const formatted = formatHandResults({
      landmarks: [[{ x: 0.1 }, null, { x: 0.2, y: 0.4, z: 0.1 }]],
    });
    expect(formatted[0].label).toBe("Unknown");
    expect(formatted[0].landmarks[0]).toEqual({ x: 0.1, y: 0, z: 0 });
    expect(formatted[0].landmarks[1]).toEqual({ x: 0, y: 0, z: 0 });
  });

  it("summarizes zero, one, and two hands", () => {
    expect(summarizeHands([])).toBe("No hands detected");
    expect(summarizeHands([{ label: "Left" }])).toBe("Left hand");
    expect(summarizeHands([{ label: "Left" }, { label: "Right" }])).toBe("Left + Right hands");
  });

  it("normalizes handedness labels", () => {
    expect(normalizeHandLabel("left")).toBe("Left");
    expect(normalizeHandLabel("RIGHT")).toBe("Right");
    expect(normalizeHandLabel("")).toBe("Unknown");
  });

  it("keeps a complete 21-point skeleton graph", () => {
    expect(HAND_CONNECTIONS.length).toBeGreaterThanOrEqual(20);
    expect(HAND_CONNECTIONS.every((pair) => pair.length === 2)).toBe(true);
    expect(Math.max(...HAND_CONNECTIONS.flat())).toBe(20);
  });
});

describe("landmark projection", () => {
  it("maps normalized coordinates onto the video pixel size", () => {
    expect(toPixel({ x: 0.25, y: 0.5 }, 640, 480)).toEqual({ x: 160, y: 240 });
  });

  it("returns null for invalid landmarks instead of throwing", () => {
    expect(toPixel(null, 640, 480)).toBeNull();
    expect(toPixel({ x: 0.2, y: 0.2 }, 0, 480)).toBeNull();
  });
});
