import { useEffect, useMemo, useState } from "react";
import { getHealth, getLabels } from "../services/api";
import { speak, stopSpeaking, isSpeechSupported } from "../utils/speech";
import { appendFinalizedHistory, finalizedText, liveSentence } from "../utils/history";
import { useCamera } from "../hooks/useCamera";
import { useHandLandmarks } from "../hooks/useHandLandmarks";
import { useTranslateStream } from "../hooks/useTranslateStream";
import CameraView from "../components/CameraView";
import PredictionCard from "../components/PredictionCard";
import ConfidenceBar from "../components/ConfidenceBar";
import ControlPanel from "../components/ControlPanel";
import Transcript from "../components/Transcript";
import StatusIndicator from "../components/StatusIndicator";

function modelStatusLabel(health) {
  if (!health) return "Unknown";
  if (health.model_mode === "mock") return "Demo Model";
  if (health.model_loaded) return "Trained Model";
  return "Model missing";
}

function modelBadge(health) {
  if (!health) return "unknown";
  if (health.model_mode === "mock") return "demo";
  if (health.model_loaded) return "real";
  return "missing";
}

function cameraBadge(camera, landmarks) {
  switch (camera.status) {
    case "live":
      return {
        status: "live",
        message:
          landmarks.status === "running"
            ? `Live · ${landmarks.fps} FPS`
            : landmarks.status === "loading"
              ? "Live · loading landmarks"
              : "Camera live",
      };
    case "requesting":
      return { status: "checking", message: "Asking for camera permission…" };
    case "denied":
      return { status: "disconnected", message: "Permission denied" };
    case "unavailable":
      return { status: "disconnected", message: camera.error || "Camera unavailable" };
    case "error":
      return { status: "disconnected", message: camera.error || "Camera error" };
    case "stopped":
      return { status: "idle", message: "Camera stopped" };
    default:
      return { status: "idle", message: "Camera idle" };
  }
}

function connectionBadge(backendState, backendMessage, stream, cameraLive) {
  if (backendState === "disconnected") {
    return { status: "disconnected", message: "Backend unavailable" };
  }
  if (backendState === "checking") {
    return { status: "checking", message: "Checking backend…" };
  }
  if (cameraLive && stream.status === "disconnected") {
    return { status: "disconnected", message: stream.message || "Stream disconnected" };
  }
  if (cameraLive && stream.status === "checking") {
    return { status: "checking", message: "Connecting stream…" };
  }
  if (cameraLive && stream.status === "live") {
    return { status: "connected", message: "Connected · live stream" };
  }
  return { status: "connected", message: backendMessage || "Backend connected" };
}

export default function Translator() {
  const [health, setHealth] = useState(null);
  const [backendState, setBackendState] = useState("checking");
  const [backendMessage, setBackendMessage] = useState("Checking backend…");
  const [labels, setLabels] = useState([]);
  const [history, setHistory] = useState([]);
  const [speechMessage, setSpeechMessage] = useState("");

  const camera = useCamera();
  const landmarks = useHandLandmarks(camera.videoRef, camera.status === "live");
  const stream = useTranslateStream({
    enabled: camera.status === "live",
    handsRef: landmarks.handsRef,
  });
  const cameraUi = cameraBadge(camera, landmarks);
  const connectionUi = connectionBadge(
    backendState,
    backendMessage,
    stream,
    camera.status === "live"
  );

  const prediction = stream.prediction || {};
  const currentSign = prediction.sign || null;
  const signDisplay = useMemo(() => {
    if (!currentSign) return null;
    const match = labels.find((item) => item.id === currentSign);
    return match?.display || currentSign;
  }, [currentSign, labels]);
  const speakText = finalizedText(prediction);
  const translation = liveSentence(prediction);
  const glosses = Array.isArray(prediction.glosses) ? prediction.glosses : [];
  const displayGlosses =
    glosses.length > 0
      ? glosses
      : Array.isArray(prediction.last_glosses)
        ? prediction.last_glosses
        : [];
  const confidence = Number.isFinite(prediction.confidence) ? prediction.confidence : 0;
  const speechReady = isSpeechSupported();
  const threshold = health?.confidence_threshold ?? 0.8;
  const cameraLive = camera.status === "live";
  const detectorRunning = landmarks.status === "running";
  const noHands = cameraLive && detectorRunning && landmarks.hands.length === 0;
  const lowConfidence =
    Boolean(currentSign) &&
    prediction.ready === true &&
    (prediction.accepted === false || (confidence > 0 && confidence < threshold));

  const finalizedFlag = prediction.finalized === true;
  const lastSentence = prediction.last_sentence || "";
  const lastGlossesKey = (prediction.last_glosses || []).join("|");
  const finalizeReason = prediction.finalize_reason || "";

  useEffect(() => {
    setHistory((items) =>
      appendFinalizedHistory(items, {
        finalized: finalizedFlag,
        last_sentence: lastSentence,
        sentence: lastSentence,
        last_glosses: lastGlossesKey ? lastGlossesKey.split("|") : [],
        finalize_reason: finalizeReason,
      })
    );
  }, [finalizedFlag, lastSentence, lastGlossesKey, finalizeReason]);

  useEffect(() => {
    let cancelled = false;

    async function refreshStatus() {
      try {
        const payload = await getHealth();
        if (cancelled) return;
        setHealth(payload);
        setBackendState("connected");
        setBackendMessage(
          payload.model_mode === "mock"
            ? "Backend connected · Demo Model"
            : payload.model_loaded
              ? "Backend connected · live inference"
              : "Backend connected · model not loaded"
        );
      } catch {
        if (cancelled) return;
        setHealth(null);
        setBackendState("disconnected");
        setBackendMessage("Backend unavailable. Start the FastAPI server on port 8000.");
      }
    }

    async function loadLabels() {
      try {
        const catalog = await getLabels();
        if (!cancelled) setLabels(catalog.labels || []);
      } catch {
        if (!cancelled) setLabels([]);
      }
    }

    refreshStatus();
    loadLabels();
    const timer = setInterval(refreshStatus, 8000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  function handleSpeak() {
    if (!speakText) {
      setSpeechMessage("Speak is only for a finalized sentence. Pause, hold, or press Finalize first.");
      return;
    }
    const ok = speak(speakText);
    setSpeechMessage(ok ? `Speaking: ${speakText}` : "Speech synthesis is not available in this browser.");
  }

  function handleFinalize() {
    stream.send?.({ type: "finalize" });
  }

  function handleClear() {
    stopSpeaking();
    stream.send?.({ type: "clear" });
    setHistory([]);
    setSpeechMessage("");
  }

  return (
    <div className="page">
      <a className="skip-link" href="#main">
        Skip to translator
      </a>

      <header className="hero">
        <p className="eyebrow">Hackathon prototype · limited ISL vocabulary</p>
        <h1>ISL Bridge</h1>
        <p className="lede">Limited-vocabulary Indian Sign Language prototype</p>
        <p className="hero-note">
          Webcam stays in the browser. Landmarks are sent to the local API. Video is
          never recorded. Speak reads a <strong>finalized</strong> sentence only when
          you click the button.
        </p>
        <StatusIndicator
          cameraStatus={cameraUi.status}
          cameraMessage={cameraUi.message}
          connectionStatus={connectionUi.status}
          connectionMessage={connectionUi.message}
          modelStatus={modelBadge(health)}
          modelMessage={modelStatusLabel(health)}
        />
      </header>

      <main id="main" className="layout">
        <section className="panel" aria-labelledby="camera-heading">
          <h2 id="camera-heading">Live camera</h2>
          <CameraView
            videoRef={camera.videoRef}
            cameraStatus={camera.status}
            cameraError={camera.error}
            onStart={camera.start}
            onStop={camera.stop}
            hands={landmarks.hands}
            handsRef={landmarks.handsRef}
            fps={landmarks.fps}
            detectorStatus={landmarks.status}
            detectorError={landmarks.error}
          />
        </section>

        <section className="panel translation-panel" aria-labelledby="prediction-heading">
          <h2 id="prediction-heading">Translation</h2>
          <PredictionCard
            sign={currentSign}
            signDisplay={signDisplay}
            sentence={translation}
            accepted={prediction.accepted === true}
            glosses={displayGlosses}
            finalized={prediction.finalized === true}
            lowConfidence={lowConfidence}
            noHands={noHands}
            message={
              prediction.type === "error" || (prediction.type === "hello" && prediction.ready === false)
                ? prediction.message
                : prediction.finalized
                  ? `Ready to speak · finalized by ${prediction.finalize_reason || "explicit"}`
                  : null
            }
            source={prediction.source}
            ready={prediction.ready}
            bufferFrames={prediction.buffer_frames}
            sequenceLength={prediction.sequence_length}
          />
          <ConfidenceBar value={confidence} threshold={threshold} low={lowConfidence} />
          <ControlPanel
            onSpeak={handleSpeak}
            onClear={handleClear}
            onFinalize={handleFinalize}
            speakDisabled={!speakText}
            finalizeDisabled={glosses.length === 0}
            clearDisabled={
              history.length === 0 &&
              !speechMessage &&
              glosses.length === 0 &&
              !prediction.last_sentence
            }
          />
          {speechMessage ? (
            <p className="hint" role="status">
              {speechMessage}
            </p>
          ) : (
            <p className="hint">
              Predictions are not spoken automatically. Finalize, then press Speak.
            </p>
          )}
          {!speechReady ? (
            <p className="alert" role="status">
              This browser does not support SpeechSynthesis.
            </p>
          ) : null}
        </section>
      </main>

      <Transcript items={history} />

      <section className="panel vocab" aria-labelledby="vocab-heading">
        <div className="panel-heading">
          <h2 id="vocab-heading">Supported signs</h2>
          <p className="panel-meta">{labels.length || "—"} in this MVP</p>
        </div>
        {labels.length === 0 ? (
          <p className="hint">Labels load from the backend config file when the API is running.</p>
        ) : (
          <ul className="vocab-list">
            {labels.map((label) => (
              <li key={label.id} title={label.hint || undefined}>
                {label.display}
              </li>
            ))}
          </ul>
        )}
        {labels.some((label) => label.hint) ? (
          <div className="sign-hints">
            <h3>How to sign the trained words</h3>
            <p className="hint">
              The live model is trained on these 10 simple motions. Match the
              movement closely. This is not a full ISL vocabulary.
            </p>
            <ul>
              {labels
                .filter((label) => label.hint)
                .map((label) => (
                  <li key={`${label.id}-hint`}>
                    <strong>{label.display}:</strong> {label.hint}
                  </li>
                ))}
            </ul>
          </div>
        ) : null}
      </section>
    </div>
  );
}
