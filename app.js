const CLASS_INFO = {
  "a primarily analog or mechanical device operated by direct physical mechanisms, analog gauges, springs, gears, valves, or continuous controls": {
    name: "Analog / Mechanical",
    descriptor: "Visible cues most closely match direct mechanical action or continuous analog operation.",
    explanation: "The image is visually closer to devices characterized by mechanical movement, direct physical controls, analog gauges, or continuous mechanisms. A rear, label, or interior view may still reveal electronic control that is not visible here.",
  },
  "a digital or electronic device whose main operation uses digital logic, electronic controls, a digital display, or digital media": {
    name: "Digital / Electronic",
    descriptor: "Visible cues most closely match electronic control, digital media, or digital interfaces.",
    explanation: "The image is visually closer to devices that use digital logic, electronic controls, digital displays, or digital media. This does not prove the exact circuitry or rule out analog components.",
  },
  "a software-controlled smart device with firmware, programmable menus, networking, applications, or embedded computing": {
    name: "Software Controlled",
    descriptor: "Visible cues most closely match programmable, connected, or embedded-computing behavior.",
    explanation: "The image is visually closer to devices associated with firmware, programmable menus, networking, application control, or embedded computing. Software cannot be directly observed from an exterior image, so this remains an inference.",
  },
  "a hybrid device combining substantial mechanical or analog functions with digital electronics or software control": {
    name: "Hybrid",
    descriptor: "The image appears to combine meaningful mechanical or analog features with electronic or software control.",
    explanation: "The visible design appears to mix physical or analog mechanisms with digital electronics or software-oriented controls. Hybrid is often the most honest category for modern appliances and audio equipment.",
  },
};

const MAX_FILE_BYTES = 12 * 1024 * 1024;
const UNCERTAINTY_TOP_SCORE = 0.39;
const UNCERTAINTY_MARGIN = 0.075;

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
const scoreList = document.getElementById("scoreList");
const resultExplanation = document.getElementById("resultExplanation");

let selectedDataUrl = null;
let worker = null;
let isRunning = false;

function ensureWorker() {
  if (worker) return worker;

  worker = new Worker(new URL("./model-worker.js", import.meta.url), { type: "module" });
  worker.addEventListener("message", handleWorkerMessage);
  worker.addEventListener("error", (event) => {
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

function renderResult(rawResults) {
  const results = Array.isArray(rawResults)
    ? rawResults
        .filter((item) => item && CLASS_INFO[item.label] && Number.isFinite(Number(item.score)))
        .map((item) => ({ ...item, score: Number(item.score) }))
        .sort((a, b) => b.score - a.score)
    : [];

  if (!results.length) {
    showError("The model returned no usable classification scores.");
    return;
  }

  const top = results[0];
  const second = results[1] ?? { score: 0 };
  const margin = top.score - second.score;
  const uncertain = top.score < UNCERTAINTY_TOP_SCORE || margin < UNCERTAINTY_MARGIN;
  const info = CLASS_INFO[top.label];

  if (uncertain) {
    resultClass.textContent = "Indeterminate";
    resultDescriptor.textContent = "The baseline did not find a sufficiently clear separation between the leading visual interpretations.";
    resultScore.textContent = `${Math.round(top.score * 100)}%`;
    resultExplanation.textContent = `The strongest visual match was ${info.name.toLowerCase()}, but the evidence was not distinct enough for this baseline to present that interpretation as its final class. Try a closer image of the controls, display, rear ports, rating label, or operating interface.`;
  } else {
    resultClass.textContent = info.name;
    resultDescriptor.textContent = info.descriptor;
    resultScore.textContent = `${Math.round(top.score * 100)}%`;
    resultExplanation.textContent = info.explanation;
  }

  scoreList.innerHTML = "";
  for (const result of results) {
    const row = document.createElement("div");
    row.className = "score-row";

    const name = document.createElement("strong");
    name.textContent = CLASS_INFO[result.label].name;

    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = `${Math.max(2, result.score * 100)}%`;
    track.appendChild(fill);

    const value = document.createElement("span");
    value.textContent = `${Math.round(result.score * 100)}%`;

    row.append(name, track, value);
    scoreList.appendChild(row);
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
  errorMessage.textContent = message;
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
