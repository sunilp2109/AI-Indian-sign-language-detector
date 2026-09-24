import { describe, expect, it } from "vitest";
import { appendFinalizedHistory, finalizedText, historySignature, liveSentence } from "../../frontend/src/utils/history.js";

describe("finalized history", () => {
  it("ignores live drafts and empty payloads", () => {
    expect(finalizedText({ sentence: "Hello.", finalized: false })).toBe("");
    expect(liveSentence({ draft_sentence: "I need." })).toBe("I need.");
    expect(appendFinalizedHistory([], { finalized: false, sentence: "Hello." })).toEqual([]);
    expect(appendFinalizedHistory([], null)).toEqual([]);
  });

  it("keeps Speak on the last finalized sentence without hiding a new draft", () => {
    const prediction = {
      finalized: false,
      last_sentence: "Hello.",
      draft_sentence: "I need a doctor.",
      sentence: "I need a doctor.",
    };
    expect(finalizedText(prediction)).toBe("Hello.");
    expect(liveSentence(prediction)).toBe("I need a doctor.");
  });

  it("stores a finalized sentence once", () => {
    const first = appendFinalizedHistory([], {
      finalized: true,
      last_sentence: "I need a doctor.",
      last_glosses: ["I", "NEED", "DOCTOR"],
      finalize_reason: "pause",
    });
    expect(first).toHaveLength(1);
    expect(first[0].text).toBe("I need a doctor.");
    expect(first[0].glosses).toEqual(["I", "NEED", "DOCTOR"]);
    expect(first[0].reason).toBe("pause");

    const again = appendFinalizedHistory(first, {
      finalized: true,
      last_sentence: "I need a doctor.",
      last_glosses: ["I", "NEED", "DOCTOR"],
      finalize_reason: "pause",
    });
    expect(again).toBe(first);
  });

  it("appends a new finalized sentence", () => {
    const start = [
      {
        text: "Thank you.",
        glosses: ["THANK_YOU"],
      },
    ];
    const next = appendFinalizedHistory(start, {
      finalized: true,
      last_sentence: "Hello.",
      last_glosses: ["HELLO"],
    });
    expect(next).toHaveLength(2);
    expect(next[0].text).toBe("Hello.");
    expect(historySignature("Hello.", ["HELLO"])).toBe("Hello.::HELLO");
  });
});
