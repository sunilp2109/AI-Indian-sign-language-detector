import { HAND_CONNECTIONS } from "./handFormat";

const COLORS = {
  Left: "#7dd3c0",
  Right: "#f0a36b",
  Unknown: "#fffaf2",
};

export function toPixel(landmark, width, height) {
  if (!landmark || !Number.isFinite(landmark.x) || !Number.isFinite(landmark.y)) {
    return null;
  }
  if (!width || !height) {
    return null;
  }
  return {
    x: landmark.x * width,
    y: landmark.y * height,
  };
}

export function drawHands(ctx, hands, width, height) {
  ctx.clearRect(0, 0, width, height);
  if (!Array.isArray(hands) || hands.length === 0) {
    return;
  }

  for (const hand of hands) {
    const color = COLORS[hand.label] || COLORS.Unknown;
    const points = (hand.landmarks || [])
      .map((landmark) => toPixel(landmark, width, height))
      .filter(Boolean);

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(2, width / 320);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    for (const [start, end] of HAND_CONNECTIONS) {
      const a = points[start];
      const b = points[end];
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, Math.max(3, width / 280), 0, Math.PI * 2);
      ctx.fill();
    }

    const wrist = points[0];
    if (wrist) {
      ctx.font = `700 ${Math.max(12, width / 32)}px Atkinson Hyperlegible, Segoe UI, sans-serif`;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.65)";
      ctx.lineWidth = 4;
      ctx.strokeText(hand.label, wrist.x + 8, wrist.y - 8);
      ctx.fillStyle = color;
      ctx.fillText(hand.label, wrist.x + 8, wrist.y - 8);
    }
  }
}
