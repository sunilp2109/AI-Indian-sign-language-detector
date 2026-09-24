/**
 * Translation history helpers. Only finalized sentences are stored.
 * Frame-level predictions must never enter this list.
 */

const MAX_HISTORY = 25;

export function finalizedText(prediction) {
  if (!prediction) return "";
  const text = (prediction.last_sentence || (prediction.finalized ? prediction.sentence : "") || "").trim();
  return text;
}

export function liveSentence(prediction) {
  if (!prediction) return "";
  return String(prediction.draft_sentence || prediction.sentence || "").trim();
}

export function historySignature(text, glosses = []) {
  return `${text}::${(glosses || []).join("|")}`;
}

export function appendFinalizedHistory(items, prediction, now = new Date()) {
  if (!prediction?.finalized) {
    return items;
  }
  const text = finalizedText(prediction);
  if (!text) {
    return items;
  }
  const glosses = Array.isArray(prediction.last_glosses)
    ? prediction.last_glosses
    : Array.isArray(prediction.glosses)
      ? prediction.glosses
      : [];
  const signature = historySignature(text, glosses);
  const previous = items[0];
  if (previous && historySignature(previous.text, previous.glosses) === signature) {
    return items;
  }
  const entry = {
    id: `${now.getTime()}-${items.length}`,
    time: now.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    text,
    glosses,
    reason: prediction.finalize_reason || "explicit",
  };
  return [entry, ...items].slice(0, MAX_HISTORY);
}
