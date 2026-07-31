import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "Xenova/clip-vit-base-patch32";
const PROMPTS = [
  "a primarily analog or mechanical device operated by direct physical mechanisms, analog gauges, springs, gears, valves, or continuous controls",
  "a digital or electronic device whose main operation uses digital logic, electronic controls, a digital display, or digital media",
  "a software-controlled smart device with firmware, programmable menus, networking, applications, or embedded computing",
  "a hybrid device combining substantial mechanical or analog functions with digital electronics or software control",
];

let classifierPromise;

function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = pipeline(
      "zero-shot-image-classification",
      MODEL_ID,
      {
        dtype: "q8",
        progress_callback: (progress) => {
          self.postMessage({ type: "progress", payload: progress });
        },
      },
    );
  }
  return classifierPromise;
}

self.addEventListener("message", async (event) => {
  const { type, image } = event.data ?? {};
  if (type !== "analyze" || !image) return;

  try {
    self.postMessage({ type: "status", payload: "Loading visual model..." });
    const classifier = await getClassifier();
    self.postMessage({ type: "status", payload: "Comparing visual evidence..." });

    const output = await classifier(image, PROMPTS, {
      hypothesis_template: "This is a photograph of {}.",
    });

    self.postMessage({ type: "result", payload: output });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: error instanceof Error ? error.message : String(error),
    });
  }
});
