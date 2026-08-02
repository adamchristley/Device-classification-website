import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "Xenova/clip-vit-base-patch32";
const MODEL_VERSION = "independent-attribute-baseline-v0.3";

const THRESHOLDS = {
  device: 0.56,
  attributePresent: 0.60,
  minimumArchitectureEvidence: 0.55,
  highEvidence: 0.72,
  moderateEvidence: 0.62,
};

const PROMPT_GROUPS = {
  device: {
    name: "Physical device",
    positive: [
      "a photograph of a physical device, machine, appliance, instrument, or electronic product",
      "a manufactured functional object with controls, ports, moving parts, a display, or operating components",
      "a clearly visible tool, machine, consumer electronic device, or household appliance",
    ],
    negative: [
      "a person, animal, plant, landscape, food item, or natural scene",
      "clothing, furniture, decoration, artwork, packaging, text, or a document rather than a functional device",
      "an abstract, blurry, cropped, or unidentifiable image without a clear physical device",
    ],
  },
  mechanical: {
    name: "Mechanical evidence",
    positive: [
      "a device whose operation depends on gears, springs, levers, valves, linkages, motors, or moving mechanisms",
      "a machine with a meaningful physical mechanism, moving media, rotating parts, a transport system, or direct mechanical action",
      "a device with mechanical controls or moving components essential to its main function",
    ],
    negative: [
      "a solid-state object with no meaningful moving mechanism or mechanical operation",
      "a device whose visible function is primarily electronic with no important moving parts",
      "a simple fixed enclosure or screen without visible mechanical action",
    ],
  },
  analog: {
    name: "Analog evidence",
    positive: [
      "an analog device using a needle, dial, gauge, continuous knob, physical scale, or continuously varying signal",
      "a device with analog controls, analog measurement, analog audio circuitry, or continuous physical indication",
      "a traditional instrument whose state is represented continuously rather than as discrete digital values",
    ],
    negative: [
      "a fully digital interface using discrete numbers, icons, menus, or binary electronic states",
      "a device with no visible analog gauge, continuous scale, analog signal control, or analog indication",
      "a purely digital electronic product rather than an analog instrument",
    ],
  },
  digital: {
    name: "Digital evidence",
    positive: [
      "a digital electronic device with an LCD, LED display, keypad, digital media, logic circuitry, or electronic controls",
      "a device that processes information in discrete digital form",
      "consumer electronics with a numeric display, digital buttons, digital storage, or digital signal processing",
    ],
    negative: [
      "a non-electronic manual mechanism with no digital display, digital media, or digital controls",
      "a purely mechanical or continuously analog instrument rather than a digital electronic device",
      "a traditional device that operates without digital logic or discrete electronic information processing",
    ],
  },
  software: {
    name: "Software-control evidence",
    positive: [
      "a programmable device controlled by software, firmware, menus, applications, networking, or an embedded computer",
      "a smart or connected device whose behavior can be configured through software",
      "a device with a touchscreen, operating system, app control, network connection, or programmable interface",
    ],
    negative: [
      "a fixed-function device with no visible programmable interface, software settings, networking, or application control",
      "a simple manual or electronic product that does not appear configurable by software",
      "a device whose behavior is determined by direct controls rather than programs, menus, firmware settings, or apps",
    ],
  },
};

const PROMPT_ENTRIES = Object.entries(PROMPT_GROUPS).flatMap(([attribute, group]) => [
  ...group.positive.map((prompt) => ({ attribute, polarity: "positive", prompt })),
  ...group.negative.map((prompt) => ({ attribute, polarity: "negative", prompt })),
]);
const PROMPTS = PROMPT_ENTRIES.map(({ prompt }) => prompt);
const PROMPT_LOOKUP = new Map(PROMPT_ENTRIES.map((entry) => [entry.prompt, entry]));

let classifierPromise;

function progressCallback(progress) {
  self.postMessage({ type: "progress", payload: progress });
}

function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = pipeline("zero-shot-image-classification", MODEL_ID, {
      dtype: "q8",
      progress_callback: progressCallback,
    });
  }
  return classifierPromise;
}

function aggregateIndependentEvidence(rawOutput) {
  const totals = Object.fromEntries(
    Object.keys(PROMPT_GROUPS).map((attribute) => [attribute, { positive: 0, negative: 0 }]),
  );

  for (const item of rawOutput ?? []) {
    const entry = PROMPT_LOOKUP.get(item?.label);
    const score = Number(item?.score);
    if (!entry || !Number.isFinite(score)) continue;
    totals[entry.attribute][entry.polarity] += score;
  }

  return Object.entries(totals).map(([id, values]) => {
    const denominator = values.positive + values.negative;
    const score = denominator > 0 ? values.positive / denominator : 0.5;
    return {
      id,
      name: PROMPT_GROUPS[id].name,
      score,
      positiveEvidence: values.positive,
      negativeEvidence: values.negative,
    };
  });
}

function evidenceLevel(score) {
  if (score >= THRESHOLDS.highEvidence) return "High";
  if (score >= THRESHOLDS.moderateEvidence) return "Moderate";
  return "Low";
}

function deriveAssessment(attributes) {
  const scores = Object.fromEntries(attributes.map(({ id, score }) => [id, score]));
  const devicePassed = scores.device >= THRESHOLDS.device;
  const architecture = {
    mechanical: scores.mechanical,
    analog: scores.analog,
    digital: scores.digital,
    software: scores.software,
  };
  const strongestArchitecture = Math.max(...Object.values(architecture));
  const physicalEvidence = Math.max(architecture.mechanical, architecture.analog);
  const electronicEvidence = Math.max(architecture.digital, architecture.software);
  const mechanicalPresent = architecture.mechanical >= THRESHOLDS.attributePresent;
  const analogPresent = architecture.analog >= THRESHOLDS.attributePresent;
  const digitalPresent = architecture.digital >= THRESHOLDS.attributePresent;
  const softwarePresent = architecture.software >= THRESHOLDS.attributePresent;

  if (!devicePassed) {
    return {
      label: "indeterminate",
      reason: "device-gate",
      evidenceScore: scores.device,
      evidenceLevel: evidenceLevel(scores.device),
      activeAttributes: [],
    };
  }

  if (strongestArchitecture < THRESHOLDS.minimumArchitectureEvidence) {
    return {
      label: "indeterminate",
      reason: "insufficient-architecture-evidence",
      evidenceScore: strongestArchitecture,
      evidenceLevel: evidenceLevel(strongestArchitecture),
      activeAttributes: [],
    };
  }

  const activeAttributes = Object.entries(architecture)
    .filter(([, score]) => score >= THRESHOLDS.attributePresent)
    .map(([id]) => id);

  const hybridEvidence = Math.min(physicalEvidence, electronicEvidence);
  if ((mechanicalPresent || analogPresent) && (digitalPresent || softwarePresent)) {
    return {
      label: "hybrid",
      reason: "physical-and-electronic-evidence",
      evidenceScore: hybridEvidence,
      evidenceLevel: evidenceLevel(hybridEvidence),
      activeAttributes,
    };
  }

  if (softwarePresent && !mechanicalPresent && !analogPresent) {
    return {
      label: "software_controlled",
      reason: "software-dominant",
      evidenceScore: architecture.software,
      evidenceLevel: evidenceLevel(architecture.software),
      activeAttributes,
    };
  }

  if (digitalPresent && !mechanicalPresent && !analogPresent) {
    return {
      label: "digital_electronic",
      reason: "digital-dominant",
      evidenceScore: architecture.digital,
      evidenceLevel: evidenceLevel(architecture.digital),
      activeAttributes,
    };
  }

  if ((mechanicalPresent || analogPresent) && !digitalPresent && !softwarePresent) {
    return {
      label: "analog_mechanical",
      reason: "physical-or-analog-dominant",
      evidenceScore: physicalEvidence,
      evidenceLevel: evidenceLevel(physicalEvidence),
      activeAttributes,
    };
  }

  return {
    label: "indeterminate",
    reason: "borderline-or-conflicting-evidence",
    evidenceScore: strongestArchitecture,
    evidenceLevel: evidenceLevel(strongestArchitecture),
    activeAttributes,
  };
}

self.addEventListener("message", async (event) => {
  const { type, image } = event.data ?? {};
  if (type !== "analyze" || !image) return;

  try {
    self.postMessage({ type: "status", payload: "Loading CLIP evidence model..." });
    const classifier = await getClassifier();
    self.postMessage({ type: "status", payload: "Testing device and architecture evidence independently..." });

    const rawOutput = await classifier(image, PROMPTS, {
      hypothesis_template: "This image shows {}.",
    });
    const attributes = aggregateIndependentEvidence(rawOutput);
    const assessment = deriveAssessment(attributes);
    const deviceAttribute = attributes.find(({ id }) => id === "device");

    self.postMessage({
      type: "result",
      payload: {
        assessment,
        deviceGate: {
          score: deviceAttribute?.score ?? 0.5,
          threshold: THRESHOLDS.device,
          passed: (deviceAttribute?.score ?? 0.5) >= THRESHOLDS.device,
        },
        attributes: attributes.filter(({ id }) => id !== "device"),
        thresholds: THRESHOLDS,
        model: {
          id: MODEL_ID,
          version: MODEL_VERSION,
          method: "Independent positive-versus-negative CLIP evidence axes with a device gate and rule-derived label",
          scoresAreIndependent: true,
          scoresSumToOne: false,
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
