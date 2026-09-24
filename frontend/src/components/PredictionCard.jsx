export default function PredictionCard({
  sign,
  signDisplay,
  sentence,
  accepted,
  message,
  source,
  ready,
  bufferFrames,
  sequenceLength,
  glosses = [],
  finalized = false,
  lowConfidence = false,
  noHands = false,
}) {
  const filling =
    ready === false &&
    Number.isFinite(bufferFrames) &&
    Number.isFinite(sequenceLength) &&
    sequenceLength > 0;
  const shownSign = signDisplay || sign;

  return (
    <article className="prediction-card" aria-live="polite">
      <p className="label">Current sign</p>
      <p className="sign">{shownSign || "—"}</p>
      {sign && signDisplay && sign !== signDisplay ? (
        <p className="sign-id">{sign}</p>
      ) : null}

      <p className="label">Gloss sequence</p>
      {glosses.length > 0 ? (
        <ol className="gloss-chips">
          {glosses.map((gloss, index) => (
            <li key={`${gloss}-${index}`}>{gloss}</li>
          ))}
        </ol>
      ) : (
        <p className="placeholder-copy">No glosses yet.</p>
      )}

      <p className="label">{finalized ? "Finalized sentence" : "Current sentence"}</p>
      <p className={`sentence ${finalized ? "sentence-final" : ""}`}>
        {sentence ? `“${sentence}”` : "Waiting for a confident sign."}
      </p>

      {filling && !sign ? (
        <p className="hint">
          Collecting frames {bufferFrames}/{sequenceLength}
        </p>
      ) : null}

      {noHands ? (
        <p className="alert" role="status">
          No hands detected — show one or both hands clearly to the camera.
        </p>
      ) : null}

      {lowConfidence ? (
        <p className="alert" role="status">
          Low confidence — please sign again, slower and in frame.
        </p>
      ) : null}

      {source === "mock" ? (
        <p className="hint">Demo Model — not a trained accuracy claim.</p>
      ) : null}

      {message ? <p className="hint">{message}</p> : null}

      {accepted === false && sign && !lowConfidence ? (
        <p className="alert" role="status">
          Uncertain — please sign again.
        </p>
      ) : null}
    </article>
  );
}
