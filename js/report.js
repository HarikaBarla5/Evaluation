// Renders the report stored in sessionStorage by interview.js.
// Falls back to demo data if none is present (so the page is viewable standalone).

document.addEventListener("DOMContentLoaded", () => {
  let report = null;
  let questions = null;

  try {
    const raw = sessionStorage.getItem("signal_report");
    if (raw) report = JSON.parse(raw);
    const rawQ = sessionStorage.getItem("signal_questions");
    if (rawQ) questions = JSON.parse(rawQ);
  } catch (e) {
    console.error("Failed to parse stored report:", e);
  }

  if (!report) {
    report = demoReport();
    questions = demoQuestions();
  }

  renderReport(report, questions);
});

function renderReport(report, questions) {
  const breakdown = report.breakdown || {};
  const weights = breakdown.weights || {};

  // Final score
  document.getElementById("finalScore").textContent = report.final_score?.toFixed?.(1) ?? "—";
  document.getElementById("bandText").innerHTML =
    `Composite band: <strong>${report.band ?? "—"}</strong>`;

  // Breakdown strip
  const strip = document.getElementById("breakdownStrip");
  strip.innerHTML = "";

  const rows = [
    { label: "Expression", value: breakdown.emotion_score, weight: weights.emotion },
    { label: "Voice confidence", value: breakdown.voice_confidence_score, weight: weights.voice_confidence },
    { label: "Voice pace", value: breakdown.voice_pace_score, weight: weights.voice_pace },
    { label: "Answer quality", value: breakdown.answer_score, weight: weights.answer },
  ];

  rows.forEach((row) => {
    if (row.value === undefined) return;
    const pct = Math.round(row.weight * 100);
    strip.insertAdjacentHTML("beforeend", `
      <div class="signal-row">
        <div class="signal-meta">
          <span class="signal-name">${row.label} <span style="color: var(--muted)">(${pct}% weight)</span></span>
          <span class="signal-value mono">${row.value.toFixed(1)}</span>
        </div>
        <div class="signal-track"><div class="signal-fill" style="--w:${row.value}%"></div></div>
      </div>
    `);
  });

  // Expression card
  document.getElementById("emotionStat").textContent =
    breakdown.emotion_score !== undefined ? breakdown.emotion_score.toFixed(1) : "—";

  const emotionDetails = report.emotion_details || {};
  const frameCount = emotionDetails.frames_analyzed ?? emotionDetails.frame_results?.length ?? 0;
  document.getElementById("emotionDetail").textContent =
    frameCount > 0
      ? `Averaged across ${frameCount} sampled frames.`
      : "Reflects average composure detected across the session.";

  // Voice card
  document.getElementById("voiceStat").textContent =
    breakdown.voice_confidence_score !== undefined ? breakdown.voice_confidence_score.toFixed(1) : "—";

  const voiceDetails = report.voice_details;
  let voiceDetailText = "Confidence and pace estimated from tone and rhythm.";
  if (Array.isArray(voiceDetails) && voiceDetails.length) {
    const avgPace = voiceDetails.reduce((a, v) => a + (v.pace_score || 0), 0) / voiceDetails.length;
    voiceDetailText = `Average pace score ${avgPace.toFixed(1)} across ${voiceDetails.length} answers.`;
  }
  document.getElementById("voiceDetail").textContent = voiceDetailText;

  // Answer card
  document.getElementById("answerStat").textContent =
    breakdown.answer_score !== undefined ? breakdown.answer_score.toFixed(1) : "—";

  const answerDetails = report.answer_details || {};
  const numAnswers = answerDetails.results?.length ?? 0;
  document.getElementById("answerDetail").textContent =
    numAnswers > 0
      ? `Average semantic match across ${numAnswers} answers vs. reference responses.`
      : "Reflects semantic similarity to reference answers.";

  // Feedback list
  const feedbackList = document.getElementById("feedbackList");
  if (Array.isArray(report.feedback) && report.feedback.length) {
    feedbackList.innerHTML = report.feedback
      .map((f) => `<li>${f}</li>`)
      .join("");
  }

  // Per-question detail
  const qaContainer = document.getElementById("questionDetails");
  const results = answerDetails.results || [];
  if (results.length && questions) {
    qaContainer.innerHTML = results
      .map((r, i) => `
        <div class="qa-item">
          <div class="qa-item-head">
            <span>Q${i + 1}: ${questions[i]?.text ?? "Question"}</span>
            <span class="mono">${r.final_score?.toFixed?.(1) ?? "—"}</span>
          </div>
          <p>Similarity: ${r.similarity_score?.toFixed?.(1) ?? "—"}
            ${r.keyword_score !== null && r.keyword_score !== undefined ? ` · Keyword match: ${r.keyword_score.toFixed(1)}` : ""}
          </p>
        </div>
      `)
      .join("");
  }
}

// ---------------- DEMO FALLBACK DATA ----------------
function demoReport() {
  return {
    final_score: 83.4,
    band: "Good",
    breakdown: {
      emotion_score: 82,
      voice_confidence_score: 74,
      voice_pace_score: 88,
      answer_score: 91,
      weights: { emotion: 0.20, voice_confidence: 0.15, voice_pace: 0.10, answer: 0.55 }
    },
    feedback: [
      "Great composure and positive expressions throughout.",
      "Strong, confident vocal delivery.",
      "Your speaking pace was well balanced.",
      "Answers were relevant and well-aligned with expected responses."
    ],
    emotion_details: { average_score: 82, frames_analyzed: 24 },
    voice_details: [
      { confidence_score: 74, pace_score: 88, syllable_rate: 3.9 },
      { confidence_score: 71, pace_score: 85, syllable_rate: 3.7 },
      { confidence_score: 78, pace_score: 91, syllable_rate: 4.1 }
    ],
    answer_details: {
      average_score: 91,
      results: [
        { similarity_score: 88.2, keyword_score: 100, final_score: 91.7 },
        { similarity_score: 84.5, keyword_score: 75, final_score: 81.65 },
        { similarity_score: 92.1, keyword_score: 100, final_score: 94.5 }
      ]
    }
  };
}

function demoQuestions() {
  return [
    { text: "Tell me about a time you had to solve a difficult problem under pressure." },
    { text: "Why do you want to work in this role?" },
    { text: "Describe a time you worked effectively as part of a team." }
  ];
}