const CLASS_INFO = {
  analog_mechanical: {
    name: "Analog / Mechanical",
    descriptor: "Mechanical or continuously analog evidence is present without clear digital or software-control evidence.",
    explanation: "The model found visible support for meaningful physical mechanisms, analog indication, or continuous controls. It did not find enough digital or software evidence to derive a more electronic category.",
  },
  digital_electronic: {
    name: "Digital / Electronic",
    descriptor: "Digital electronic evidence is present without strong mechanical, analog, or software-control evidence.",
    explanation: "The model found visible support for digital displays, digital media, discrete electronic controls, or digital information processing. This does not prove the exact internal circuitry.",
  },
  software_controlled: {
    name: "Software Controlled",
    descriptor: "Programmable or software-oriented evidence is present without strong mechanical or analog evidence.",
    explanation: "The model found visible support for menus, applications, networking, embedded computing, or configurable behavior. Software cannot be directly observed from one exterior image, so this remains an inference.",
  },
  hybrid: {
    name: "Hybrid",
    descriptor: "Meaningful physical or analog evidence and digital or software evidence are both present.",
    explanation: "The final label is derived from independent evidence axes. The image appears to combine a meaningful mechanism or analog subsystem with digital electronics or software-oriented control.",
  },
  indeterminate: {
    name: "Indeterminate",
    descriptor: "The image does not provide enough clear, internally consistent architecture evidence for a derived label.",
    explanation: "The system is designed to abstain when the physical-device gate fails, architecture evidence is weak, or several axes remain borderline. Another angle, the control panel, rear ports, an interior view, or a model number may be needed.",
  },
};

const ATTRIBUTE_INFO = {
  mechanical: "Mechanical evidence",
  analog: "Analog evidence",
  digital: "Digital evidence",
  software: "Software-control evidence",
};

const MAX_FILE_BYTES = 12 * 1024 * 1024;

const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const emptyUpload = document.getElementById("emptyUpload");
const previewImage = document.getElementById("previewImage");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");
const progressWrap = document.getElementById("progressWrap");
const progressText = document.getElementById("progressText");
const progressPercent = document.getElementById("progressPercent");
const progressBar = document.getElementById("progressBar");
const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const resultError = document.getElementById("resultError");
const errorMessage = document.getElementById("errorMessage");
const resultClass = document.getElementById("resultClass");
const resultDescriptor = document.getElementById("resultDescriptor");
const resultScore = document.getElementById("resultScore");
const resultScoreLabel = document.getElementById("resultScoreLabel") || document.querySelector(".score-badge span");
const scoreList = document.getElementById("scoreList");
const resultExplanation = document.getElementById("resultExplanation");
const resultModelLabel = document.getElementById("resultModelLabel");

let selectedDataUrl = null;
let worker = null;
let isRunning = false;

function ensureWorker() {
  if (worker) return worker;

  worker = new Worker(new URL("./model-worker.js", import.meta.url), { type: "module" });
  worker.addEventListener("message", handleWorkerMessage);
  worker.addEventListener("error", (event) => {
    finishRunning();
    showError(event.message || "The browser could not start the model worker.");
  });
  return worker;
}

function handleWorkerMessage(event) {
  const { type, payload } = event.data ?? {};

  if (type === "progress") {
    updateProgress(payload);
    return;
  }

  if (type === "status") {
    progressWrap.hidden = false;
    progressText.textContent = payload;
    return;
  }

  if (type === "result") {
    finishRunning();
    renderResult(payload);
    return;
  }

  if (type === "error") {
    finishRunning();
    showError(payload || "Unknown model error.");
  }
}

function updateProgress(progress) {
  progressWrap.hidden = false;

  const status = progress?.status;
  const file = progress?.file;
  const percent = Number(progress?.progress);

  if (status === "progress" && Number.isFinite(percent)) {
    const bounded = Math.max(0, Math.min(100, percent));
    progressBar.style.width = `${bounded}%`;
    progressPercent.textContent = `${Math.round(bounded)}%`;
    progressText.textContent = file ? `Downloading ${shortFileName(file)}...` : "Downloading model...";
  } else if (status === "ready") {
    progressBar.style.width = "100%";
    progressPercent.textContent = "Ready";
    progressText.textContent = "Model loaded";
  } else if (status === "initiate") {
    progressText.textContent = file ? `Preparing ${shortFileName(file)}...` : "Preparing model...";
  }
}

function shortFileName(file) {
  const parts = String(file).split("/");
  const value = parts.at(-1) || "model file";
  return value.length > 34 ? `${value.slice(0, 31)}...` : value;
}

function processFile(file) {
  resetResult();

  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("Please choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError("Please choose an image smaller than 12 MB.");
    return;
  }

  const reader = new FileReader();
  reader.addEventListener("load", () => {
    selectedDataUrl = String(reader.result);
    previewImage.src = selectedDataUrl;
    previewImage.hidden = false;
    emptyUpload.hidden = true;
    analyzeButton.disabled = false;
    clearButton.disabled = false;
  });
  reader.addEventListener("error", () => showError("The selected image could not be read."));
  reader.readAsDataURL(file);
}

function setRunning(running) {
  isRunning = running;
  analyzeButton.disabled = running || !selectedDataUrl;
  clearButton.disabled = running || !selectedDataUrl;
  imageInput.disabled = running;
  analyzeButton.textContent = running ? "Analyzing..." : "Analyze image";
}

function finishRunning() {
  setRunning(false);
  setTimeout(() => {
    progressWrap.hidden = true;
    progressBar.style.width = "4%";
    progressPercent.textContent = "";
  }, 700);
}

function analyzeImage() {
  if (!selectedDataUrl || isRunning) return;

  resetResult();
  setRunning(true);
  progressWrap.hidden = false;
  progressText.textContent = "Preparing model...";
  progressPercent.textContent = "";
  progressBar.style.width = "4%";

  ensureWorker().postMessage({ type: "analyze", image: selectedDataUrl });
}

function interpretationFor(payload, info) {
  const reason = payload?.assessment?.reason;
  const deviceScore = Number(payload?.deviceGate?.score);
  const active = Array.isArray(payload?.assessment?.activeAttributes)
    ? payload.assessment.activeAttributes.map((id) => ATTRIBUTE_INFO[id] ?? id)
    : [];

  if (reason === "device-gate") {
    return `The image scored ${Math.round(deviceScore * 100)}% on the physical-device gate, below the required threshold. The model therefore did not force architecture labels onto an image that may not clearly show a device.`;
  }
  if (reason === "insufficient-architecture-evidence") {
    return "The image appears to show a physical device, but none of the four architecture axes produced enough evidence for a reliable derived label. A closer view of controls, displays, ports, moving parts, or labels may help.";
  }
  if (reason === "borderline-or-conflicting-evidence") {
    return "Some architecture evidence was present, but it stayed near the decision thresholds or produced a combination that the current rule set cannot resolve safely. The system returned Indeterminate instead of forcing a class.";
  }

  const evidenceSummary = active.length
    ? ` Evidence above the decision threshold: ${active.join(", ")}.`
    : "";
  return `${info.explanation}${evidenceSummary}`;
}

function addScoreRow(nameText, score, note = "") {
  const row = document.createElement("div");
  row.className = "score-row";

  const name = document.createElement("strong");
  name.textContent = note ? `${nameText} · ${note}` : nameText;

  const track = document.createElement("div");
  track.className = "score-track";
  const fill = document.createElement("div");
  fill.className = "score-fill";
  fill.style.width = `${Math.max(2, Math.min(100, score * 100))}%`;
  track.appendChild(fill);

  const value = document.createElement("span");
  value.textContent = `${Math.round(score * 100)}%`;

  row.append(name, track, value);
  scoreList.appendChild(row);
}

function renderResult(payload) {
  const assessment = payload?.assessment;
  const attributes = Array.isArray(payload?.attributes) ? payload.attributes : [];
  const model = payload?.model;
  const info = CLASS_INFO[assessment?.label];

  if (!assessment || !info || attributes.length !== 4) {
    showError("The model returned an incomplete evidence assessment.");
    return;
  }

  if (resultModelLabel) {
    resultModelLabel.textContent = model?.version
      ? `Evidence assessment · ${model.version}`
      : "Evidence assessment";
  }

  resultClass.textContent = info.name;
  resultDescriptor.textContent = info.descriptor;
  resultScore.textContent = assessment.evidenceLevel || "Low";
  if (resultScoreLabel) resultScoreLabel.textContent = "evidence level";
  resultExplanation.textContent = interpretationFor(payload, info);

  scoreList.innerHTML = "";
  addScoreRow(
    "Physical device gate",
    Number(payload?.deviceGate?.score) || 0,
    payload?.deviceGate?.passed ? "passed" : "not passed",
  );

  for (const attribute of attributes) {
    if (!ATTRIBUTE_INFO[attribute?.id] || !Number.isFinite(Number(attribute?.score))) continue;
    const threshold = Number(payload?.thresholds?.attributePresent) || 0.60;
    addScoreRow(
      ATTRIBUTE_INFO[attribute.id],
      Number(attribute.score),
      Number(attribute.score) >= threshold ? "present" : "below threshold",
    );
  }

  resultEmpty.hidden = true;
  resultError.hidden = true;
  resultContent.hidden = false;
}

function showError(message) {
  setRunning(false);
  progressWrap.hidden = true;
  resultEmpty.hidden = true;
  resultContent.hidden = true;
  resultError.hidden = false;
  errorMessage.textContent = message.includes("fetch")
    ? "The browser could not download the model. Check the connection, disable strict content blockers for this page, and try again."
    : message;
}

function resetResult() {
  resultEmpty.hidden = false;
  resultContent.hidden = true;
  resultError.hidden = true;
}

function clearImage() {
  selectedDataUrl = null;
  imageInput.value = "";
  previewImage.removeAttribute("src");
  previewImage.hidden = true;
  emptyUpload.hidden = false;
  analyzeButton.disabled = true;
  clearButton.disabled = true;
  progressWrap.hidden = true;
  resetResult();
}

imageInput.addEventListener("change", (event) => processFile(event.target.files?.[0]));
analyzeButton.addEventListener("click", analyzeImage);
clearButton.addEventListener("click", clearImage);

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!isRunning) dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  if (!isRunning) processFile(event.dataTransfer?.files?.[0]);
});
