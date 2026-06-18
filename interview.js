// ================= CONFIG =================
const API_BASE = "http://localhost:8000"; // adjust to your backend URL

// Default questions (used when no resume uploaded)
const DEFAULT_QUESTIONS = [
  {
    text: "Tell me about a time you had to solve a difficult problem under pressure.",
    ideal_answer: "Describe a specific challenging situation, the actions you took to analyze and resolve it, and the positive outcome or lesson learned.",
    keywords: ["problem", "solution", "result", "team"]
  },
  {
    text: "Why do you want to work in this role?",
    ideal_answer: "Explain genuine interest in the role and company, how your skills align with the position, and what you hope to contribute and grow into.",
    keywords: ["skills", "growth", "company", "contribute"]
  },
  {
    text: "Describe a time you worked effectively as part of a team.",
    ideal_answer: "Give an example of collaboration, your specific role within the team, how you communicated, and the successful outcome achieved together.",
    keywords: ["team", "communication", "collaboration", "outcome"]
  }
];

// Load AI-generated questions from resume page if available
let QUESTIONS = DEFAULT_QUESTIONS;
try {
  const stored = sessionStorage.getItem("signal_questions");
  if (stored) {
    const parsed = JSON.parse(stored);
    // resume.html stores {question, ideal_answer, keywords}
    // interview.js expects {text, ideal_answer, keywords} — remap
    if (parsed.length > 0 && parsed[0].question) {
      QUESTIONS = parsed.map(q => ({
        text: q.question,
        ideal_answer: q.ideal_answer,
        keywords: q.keywords || []
      }));
    } else if (parsed.length > 0 && parsed[0].text) {
      QUESTIONS = parsed; // already in right format
    }
  }
} catch (e) {
  console.warn("Could not load stored questions, using defaults.", e);
}

// ================= STATE =================
let stream = null;
let videoRecorder = null;
let videoChunks = [];
let fullVideoBlob = null;

let audioRecorder = null;
let audioChunks = [];
const answerAudioBlobs = []; // one per question
const answerTexts = [];

let currentQuestionIndex = 0;
let timerInterval = null;
let sessionStartTime = null;

// ================= DOM =================
const preview = document.getElementById("preview");
const recIndicator = document.getElementById("recIndicator");
const recLabel = document.getElementById("recLabel");
const timerEl = document.getElementById("timer");
const questionText = document.getElementById("questionText");
const answerText = document.getElementById("answerText");
const qIndexEl = document.getElementById("qIndex");
const qTotalEl = document.getElementById("qTotal");

const startBtn = document.getElementById("startBtn");
const recordBtn = document.getElementById("recordBtn");
const nextBtn = document.getElementById("nextBtn");
const finishBtn = document.getElementById("finishBtn");

const liveAnswers = document.getElementById("liveAnswers");
const fillAnswers = document.getElementById("fillAnswers");

qTotalEl.textContent = QUESTIONS.length;
questionText.textContent = QUESTIONS[0].text;

// ================= TIMER =================
function formatTime(ms) {
  const totalSec = Math.floor(ms / 1000);
  const min = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const sec = String(totalSec % 60).padStart(2, "0");
  return `${min}:${sec}`;
}

function startTimer() {
  sessionStartTime = Date.now();
  timerInterval = setInterval(() => {
    timerEl.textContent = formatTime(Date.now() - sessionStartTime);
  }, 500);
}

function stopTimer() {
  clearInterval(timerInterval);
}

// ================= SETUP (camera + mic) =================
startBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    preview.srcObject = stream;

    // Start full-session video recording (for emotion analysis)
    videoChunks = [];
    videoRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    videoRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) videoChunks.push(e.data);
    };
    videoRecorder.onstop = () => {
      fullVideoBlob = new Blob(videoChunks, { type: "video/webm" });
    };
    videoRecorder.start();

    startTimer();

    startBtn.disabled = true;
    recordBtn.disabled = false;
    recIndicator.classList.add("recording");
    recLabel.textContent = "Recording session";

  } catch (err) {
    alert("Could not access camera/microphone: " + err.message);
  }
});

// ================= PER-QUESTION AUDIO RECORDING =================
recordBtn.addEventListener("click", async () => {
  if (recordBtn.textContent.includes("Record answer")) {
    // start recording this answer's audio
    audioChunks = [];
    const audioStream = new MediaStream(stream.getAudioTracks());
    audioRecorder = new MediaRecorder(audioStream, { mimeType: "audio/webm" });
    audioRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    audioRecorder.start();

    recordBtn.textContent = "■ Stop recording";
    recordBtn.classList.add("recording");

  } else {
    // stop recording this answer's audio
    audioRecorder.stop();
    await new Promise((resolve) => {
      audioRecorder.onstop = resolve;
    });

    const blob = new Blob(audioChunks, { type: "audio/webm" });
    answerAudioBlobs[currentQuestionIndex] = blob;

    recordBtn.textContent = "● Record answer";
    recordBtn.classList.remove("recording");

    nextBtn.disabled = false;
    updateAnswerProgress();
  }
});

// ================= NAVIGATION =================
nextBtn.addEventListener("click", () => {
  // save current answer text
  answerTexts[currentQuestionIndex] = answerText.value;

  currentQuestionIndex++;

  if (currentQuestionIndex < QUESTIONS.length) {
    questionText.textContent = QUESTIONS[currentQuestionIndex].text;
    answerText.value = "";
    qIndexEl.textContent = currentQuestionIndex + 1;
    nextBtn.disabled = true;
  } else {
    nextBtn.disabled = true;
    recordBtn.disabled = true;
    finishBtn.disabled = false;
    questionText.textContent = "All questions complete. Click \"Finish & get report\" when ready.";
  }
});

function updateAnswerProgress() {
  const answered = answerAudioBlobs.filter(Boolean).length;
  liveAnswers.textContent = `${answered} / ${QUESTIONS.length}`;
  fillAnswers.style.setProperty("--w", `${(answered / QUESTIONS.length) * 100}%`);
}

// ================= FINISH SESSION =================
finishBtn.addEventListener("click", async () => {
  // save last answer text
  answerTexts[currentQuestionIndex] = answerText.value;

  finishBtn.disabled = true;
  finishBtn.textContent = "Processing...";

  // stop session recording
  videoRecorder.stop();
  stopTimer();
  recIndicator.classList.remove("recording");
  recLabel.textContent = "Processing";

  await new Promise((resolve) => {
    videoRecorder.onstop = () => {
      fullVideoBlob = new Blob(videoChunks, { type: "video/webm" });
      resolve();
    };
  });

  // stop camera
  stream.getTracks().forEach((t) => t.stop());

  try {
    const formData = new FormData();
    formData.append("video", fullVideoBlob, "session.webm");

    answerAudioBlobs.forEach((blob, i) => {
      formData.append("answers_audio", blob, `answer_${i}.webm`);
    });

    const qaData = QUESTIONS.map((q, i) => ({
      candidate_answer: answerTexts[i] || "",
      ideal_answer: q.ideal_answer,
      keywords: q.keywords
    }));
    formData.append("qa_data", JSON.stringify(qaData));
    formData.append("sample_rate", "15");

    const response = await fetch(`${API_BASE}/interview/submit`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed: ${response.status}`);
    }

    const report = await response.json();

    // store report and redirect
    sessionStorage.setItem("signal_report", JSON.stringify(report));
    sessionStorage.setItem("signal_questions", JSON.stringify(QUESTIONS));
    window.location.href = "report.html";

  } catch (err) {
    alert("Could not generate report: " + err.message);
    finishBtn.textContent = "Finish & get report";
    finishBtn.disabled = false;
  }
});