export default function ControlPanel({
  onSpeak,
  onClear,
  onFinalize,
  speakDisabled = true,
  clearDisabled = false,
  finalizeDisabled = true,
}) {
  return (
    <div className="controls" role="group" aria-label="Translation actions">
      <button type="button" onClick={onSpeak} disabled={speakDisabled}>
        Speak
      </button>
      <button type="button" className="secondary" onClick={onFinalize} disabled={finalizeDisabled}>
        Finalize
      </button>
      <button type="button" className="secondary" onClick={onClear} disabled={clearDisabled}>
        Clear
      </button>
    </div>
  );
}
