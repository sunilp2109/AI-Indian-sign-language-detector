import { describe, expect, it } from "vitest";
import { isSpeechSupported, speak } from "../../frontend/src/utils/speech.js";

describe("speech helper", () => {
  it("does not speak unless SpeechSynthesis exists", () => {
    expect(speak("")).toBe(false);
    if (!isSpeechSupported()) {
      expect(speak("Hello.")).toBe(false);
    }
  });
});
