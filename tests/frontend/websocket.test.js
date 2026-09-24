import { describe, expect, it } from "vitest";
import { serializeHands } from "../../frontend/src/services/websocket.js";

function twentyOnePoints() {
  return Array.from({ length: 21 }, (_, i) => ({ x: i * 0.01, y: 0.2, z: 0, w: 9 }));
}

describe("serializeHands", () => {
  it("keeps complete left/right landmarks and drops extra fields", () => {
    const payload = serializeHands([
      {
        label: "Right",
        score: 0.99,
        landmarks: twentyOnePoints(),
      },
    ]);
    expect(payload).toHaveLength(1);
    expect(payload[0].label).toBe("Right");
    expect(payload[0].landmarks).toHaveLength(21);
    expect(payload[0].landmarks[0]).toEqual({ x: 0, y: 0.2, z: 0 });
    expect(payload[0].landmarks[0].w).toBeUndefined();
  });

  it("drops incomplete landmark sets", () => {
    expect(
      serializeHands([{ label: "Right", landmarks: [{ x: 0.1, y: 0.2, z: 0.3 }] }])
    ).toEqual([]);
  });

  it("returns an empty list for missing hands", () => {
    expect(serializeHands(null)).toEqual([]);
    expect(serializeHands(undefined)).toEqual([]);
  });
});
