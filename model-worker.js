import {
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.useBrowserCache = true;

const BASE_MODEL_ID = "Xenova/clip-vit-base-patch32";
const FALLBACK_VERSION = "zero-shot-ensemble-v0.2";
const CLASSIFIER_URL = new URL("./models/classifier.json", self.location.href);

const CLASS_PROMPTS = {
  analog_mechanical: [
    "a mechanical device with gears, springs, levers, valves, or direct physical controls",
    "an analog instrument with a dial, needle, gauge, or continuous control",
    "a traditional non-computerized device operated by physical mechanisms",
    "a manually controlled machine without a digital display or software interface",
  ],
  digital_electronic: [
    "an electronic device with a digital display, keypad, LEDs, or digital logic",
    "consumer electronics using digital media or electronic controls",
    "a digital instrument with a screen, numeric readout, or electronic interface",
    "an electronically controlled device without obvious networking or application control",
  ],
  software_controlled: [
    "a smart connected device controlled by software, firmware, or a mobile application",
    "a programmable device with menus, a touchscreen, networking, or embedded computing",
    "an internet-connected appliance or embedded computer system",
    "a device whose behavior is configured through software settings or applications",
  ],
  hybrid: [
    "a machine combining mechanical moving parts with digital electronic control",
    "an appliance with physical mechanisms and programmable electronic controls",
    "a device with analog or mechanical functions and a digital user interface",
    "a mixed-architecture system such as a modern stereo, washer, or vehicle subsystem",
  ],
};

const PROMPT_ENTRIES = Object.entries(CLASS_PROMPTS).flatMap(([classId, prompts]) =>
  prompts.map((prompt) => ({ classId, prompt })),
);
const PROMPTS = PROMPT_ENTRIES.map(({ prompt }) => prompt);
const PROMPT_CLASS = new Map(PROMPT_ENTRIES.map(({ classId, prompt }) => [prompt, classId]));

let classifierArtifactPromise;
let visionRuntimePromise;
let zeroShotPromise;

function progressCallback(progress) {
  self.postMessage({ type: "progress", payload: progress });
}

async function loadClassifierArtifact() {
  if (!classifierArtifactPromise) {
    classifierArtifactPromise = fetch(CLASSIFIER_URL, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) return null;
        const artifact = await response.json();
        const valid = artifact?.schema_version === 1
          && artifact?.base_model === BASE_MODEL_ID
          && artifact?.embedding_dimensions === 512
          && Array.isArray(artifact?.labels)
          && artifact.labels.length === 4
          && Array.isArray(artifact?.weights)
          && artifact.weights.length === 4;
        return valid ? artifact : null;
      })
      .catch(() => null);
  }
  return classifierArtifactPromise;
}

async function getVisionRuntime() {
  if (!visionRuntimePromise) {
    visionRuntimePromise = Promise.all([
      AutoProcessor.from_pretrained(BASE_MODEL_ID, { progress_callback: progressCallback }),
      CLIPVisionModelWithProjection.from_pretrained(BASE_MODEL_ID, {
        dtype: "q8",
        progress_callback: progressCallback,
      }),
    ]).then(([processor, model]) => ({ processor, model }));
  }
  return visionRuntimePromise;
}

function getZeroShotClassifier() {
  if (!zeroShotPromise) {
    zeroShotPromise = pipeline("zero-shot-image-classification", BASE_MODEL_ID, {
      dtype: "q8",
      progress_callback: progressCallback,
    });
  }
  return zeroShotPromise;
}

function l2Normalize(values) {
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
  return values.map((value) => value / norm);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exponents = logits.map((value) => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

async function dataUrlToRawImage(dataUrl) {
  const response = await fetch(dataUrl);
  if (!response.ok) throw new Error("The selected image could not be decoded.");
  return RawImage.fromBlob(await response.blob());
}

async function classifyWithTrainedHead(image, artifact) {
  self.postMessage({ type: "status", payload: "Loading trained CLIP encoder..." });
  const { processor, model } = await getVisionRuntime();
  const rawImage = await dataUrlToRawImage(image);
  const inputs = await processor(rawImage);
  self.postMessage({ type: "status", payload: "Applying trained architecture classifier..." });
  const output = await model(inputs);
  const embedding = l2Normalize(Array.from(output.image_embeds.data));
  const logits = artifact.weights.map((row, classIndex) =>
    row.reduce((sum, weight, index) => sum + weight * embedding[index], Number(artifact.bias?.[classIndex] ?? 0)),
  );
  const probabilities = softmax(logits);
  const results = artifact.labels
    .map((label, index) => ({ label, score: probabilities[index] }))
    .sort((a, b) => b.score - a.score);

  return {
    results,
    model: {
      id: artifact.base_model,
      version: artifact.model_version,
      method: "Frozen CLIP encoder with trained logistic-regression head",
      thresholds: artifact.thresholds,
      metrics: artifact.metrics,
      dataset: artifact.dataset_version,
      warning: artifact.warning,
    },
  };
}

function aggregatePromptScores(rawOutput) {
  const totals = Object.fromEntries(Object.keys(CLASS_PROMPTS).map((classId) => [classId, 0]));
  for (const item of rawOutput ?? []) {
    const classId = PROMPT_CLASS.get(item?.label);
    const score = Number(item?.score);
    if (classId && Number.isFinite(score)) totals[classId] += score;
  }
  const totalScore = Object.values(totals).reduce((sum, value) => sum + value, 0);
  if (!(totalScore > 0)) return [];
  return Object.entries(totals)
    .map(([label, score]) => ({ label, score: score / totalScore }))
    .sort((a, b) => b.score - a.score);
}

async function classifyWithZeroShot(image) {
  self.postMessage({ type: "status", payload: "Loading CLIP zero-shot baseline..." });
  const classifier = await getZeroShotClassifier();
  self.postMessage({ type: "status", payload: "Comparing architecture evidence..." });
  const rawOutput = await classifier(image, PROMPTS, {
    hypothesis_template: "This is a photograph of {}.",
  });
  const results = aggregatePromptScores(rawOutput);
  if (!results.length) throw new Error("The model returned no usable class scores.");
  return {
    results,
    model: {
      id: BASE_MODEL_ID,
      version: FALLBACK_VERSION,
      method: "CLIP zero-shot prompt ensemble fallback",
      promptsPerClass: 4,
    },
  };
}

self.addEventListener("message", async (event) => {
  const { type, image } = event.data ?? {};
  if (type !== "analyze" || !image) return;

  try {
    const artifact = await loadClassifierArtifact();
    const payload = artifact
      ? await classifyWithTrainedHead(image, artifact)
      : await classifyWithZeroShot(image);
    self.postMessage({ type: "result", payload });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: error instanceof Error ? error.message : String(error),
    });
  }
});
