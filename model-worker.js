import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "Xenova/clip-vit-base-patch32";
const MODEL_VERSION = "zero-shot-ensemble-v0.2";

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

let classifierPromise;

function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = pipeline("zero-shot-image-classification", MODEL_ID, {
      dtype: "q8",
      progress_callback: (progress) => {
        self.postMessage({ type: "progress", payload: progress });
      },
    });
  }
  return classifierPromise;
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

self.addEventListener("message", async (event) => {
  const { type, image } = event.data ?? {};
  if (type !== "analyze" || !image) return;

  try {
    self.postMessage({ type: "status", payload: "Loading CLIP visual encoder..." });
    const classifier = await getClassifier();
    self.postMessage({ type: "status", payload: "Comparing architecture evidence..." });

    const rawOutput = await classifier(image, PROMPTS, {
      hypothesis_template: "This is a photograph of {}.",
    });
    const results = aggregatePromptScores(rawOutput);

    if (!results.length) throw new Error("The model returned no usable class scores.");

    self.postMessage({
      type: "result",
      payload: {
        results,
        model: {
          id: MODEL_ID,
          version: MODEL_VERSION,
          method: "CLIP zero-shot prompt ensemble",
          promptsPerClass: 4,
        },
      },
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: error instanceof Error ? error.message : String(error),
    });
  }
});
