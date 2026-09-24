/**
 * Browser SpeechSynthesis. Call only from an explicit user action.
 * Never speak live/unstable predictions.
 */
export function speak(text) {
  if (!text || typeof window === "undefined" || !window.speechSynthesis) {
    return false;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(String(text).trim());
  utterance.lang = "en-IN";
  utterance.rate = 0.95;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
  return true;
}

export function stopSpeaking() {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    return;
  }
  window.speechSynthesis.cancel();
}

export function isSpeechSupported() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}
