import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import crypto from 'node:crypto';
import {
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from '@huggingface/transformers';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const MANIFEST_PATH = path.join(ROOT, 'data', 'manifest.csv');
const CACHE_DIR = path.join(ROOT, 'training', '.cache', 'images');
const MODEL_PATH = path.join(ROOT, 'models', 'classifier.json');
const METRICS_PATH = path.join(ROOT, 'results', 'metrics.json');
const PREDICTIONS_PATH = path.join(ROOT, 'results', 'predictions.csv');
const MATRIX_PATH = path.join(ROOT, 'results', 'confusion-matrix.svg');
const CARD_PATH = path.join(ROOT, 'results', 'MODEL_CARD.md');
const BASE_MODEL = 'Xenova/clip-vit-base-patch32';
const LABELS = ['analog_mechanical', 'digital_electronic', 'software_controlled', 'hybrid'];
const LABEL_NAMES = {
  analog_mechanical: 'Analog / Mechanical',
  digital_electronic: 'Digital / Electronic',
  software_controlled: 'Software Controlled',
  hybrid: 'Hybrid',
};
const USER_AGENT = 'DeviceArchitectureResearch/0.1 (https://github.com/adamchristley/Device-classification-website)';
const SEED = 20260731;

env.allowLocalModels = false;
env.useBrowserCache = false;
env.useFSCache = true;
env.cacheDir = path.join(ROOT, 'training', '.cache', 'models');

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else if (ch !== '\r') field += ch;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const [headers, ...body] = rows.filter((r) => r.some((value) => value !== ''));
  return body.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ''])));
}

function csvEscape(value) {
  const text = value == null ? '' : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function l2Normalize(vector) {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / norm);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function predictOne(vector, weights, bias) {
  const logits = weights.map((row, classIndex) => row.reduce((sum, weight, i) => sum + weight * vector[i], bias[classIndex]));
  return softmax(logits);
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
  return best;
}

function shuffleInPlace(array, random) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function trainLogisticRegression(samples, dimensions) {
  const random = seededRandom(SEED);
  const weights = Array.from({ length: LABELS.length }, () => Array.from({ length: dimensions }, () => (random() - 0.5) * 0.002));
  const bias = Array(LABELS.length).fill(0);
  const mw = weights.map((row) => row.map(() => 0));
  const vw = weights.map((row) => row.map(() => 0));
  const mb = bias.map(() => 0);
  const vb = bias.map(() => 0);
  const beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, learningRate = 0.018, l2 = 0.003;
  const epochs = 420;
  const indices = samples.map((_, index) => index);
  let step = 0;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    shuffleInPlace(indices, random);
    let epochLoss = 0;
    for (const sampleIndex of indices) {
      step += 1;
      const sample = samples[sampleIndex];
      const probabilities = predictOne(sample.embedding, weights, bias);
      epochLoss += -Math.log(Math.max(1e-9, probabilities[sample.labelIndex]));

      for (let c = 0; c < LABELS.length; c += 1) {
        const error = probabilities[c] - (c === sample.labelIndex ? 1 : 0);
        const gradBias = error;
        mb[c] = beta1 * mb[c] + (1 - beta1) * gradBias;
        vb[c] = beta2 * vb[c] + (1 - beta2) * gradBias * gradBias;
        const mbHat = mb[c] / (1 - beta1 ** step);
        const vbHat = vb[c] / (1 - beta2 ** step);
        bias[c] -= learningRate * mbHat / (Math.sqrt(vbHat) + epsilon);

        for (let d = 0; d < dimensions; d += 1) {
          const gradient = error * sample.embedding[d] + l2 * weights[c][d];
          mw[c][d] = beta1 * mw[c][d] + (1 - beta1) * gradient;
          vw[c][d] = beta2 * vw[c][d] + (1 - beta2) * gradient * gradient;
          const mHat = mw[c][d] / (1 - beta1 ** step);
          const vHat = vw[c][d] / (1 - beta2 ** step);
          weights[c][d] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }
      }
    }
    if (epoch % 70 === 0 || epoch === epochs - 1) console.log(`epoch=${epoch + 1} loss=${(epochLoss / samples.length).toFixed(4)}`);
  }
  return { weights, bias, hyperparameters: { optimizer: 'Adam', epochs, learningRate, l2, seed: SEED } };
}

function confusionMatrix(samples, weights, bias) {
  const matrix = Array.from({ length: LABELS.length }, () => Array(LABELS.length).fill(0));
  const predictions = samples.map((sample) => {
    const scores = predictOne(sample.embedding, weights, bias);
    const predictedIndex = argmax(scores);
    matrix[sample.labelIndex][predictedIndex] += 1;
    return { ...sample, scores, predictedIndex };
  });
  return { matrix, predictions };
}

function classificationMetrics(predictions) {
  const perClass = {};
  for (let c = 0; c < LABELS.length; c += 1) {
    let tp = 0, fp = 0, fn = 0;
    for (const item of predictions) {
      if (item.predictedIndex === c && item.labelIndex === c) tp += 1;
      else if (item.predictedIndex === c) fp += 1;
      else if (item.labelIndex === c) fn += 1;
    }
    const precision = tp / Math.max(1, tp + fp);
    const recall = tp / Math.max(1, tp + fn);
    const f1 = 2 * precision * recall / Math.max(1e-12, precision + recall);
    perClass[LABELS[c]] = { precision, recall, f1, support: predictions.filter((item) => item.labelIndex === c).length };
  }
  const correct = predictions.filter((item) => item.predictedIndex === item.labelIndex).length;
  const macroF1 = Object.values(perClass).reduce((sum, item) => sum + item.f1, 0) / LABELS.length;
  return { accuracy: correct / Math.max(1, predictions.length), macro_f1: macroF1, per_class: perClass };
}

function tuneThresholds(predictions) {
  let best = { min_probability: 0, min_margin: 0, coverage: 1, selective_accuracy: 0, score: -Infinity };
  for (let p = 0.25; p <= 0.75; p += 0.025) {
    for (let m = 0; m <= 0.35; m += 0.025) {
      const answered = predictions.filter((item) => {
        const sorted = [...item.scores].sort((a, b) => b - a);
        return sorted[0] >= p && sorted[0] - sorted[1] >= m;
      });
      const coverage = answered.length / Math.max(1, predictions.length);
      if (coverage < 0.6) continue;
      const accuracy = answered.filter((item) => item.predictedIndex === item.labelIndex).length / Math.max(1, answered.length);
      const score = accuracy + 0.12 * coverage;
      if (score > best.score) best = { min_probability: Number(p.toFixed(3)), min_margin: Number(m.toFixed(3)), coverage, selective_accuracy: accuracy, score };
    }
  }
  delete best.score;
  return best;
}

function selectiveMetrics(predictions, thresholds) {
  const answered = predictions.filter((item) => {
    const sorted = [...item.scores].sort((a, b) => b - a);
    return sorted[0] >= thresholds.min_probability && sorted[0] - sorted[1] >= thresholds.min_margin;
  });
  return {
    coverage: answered.length / Math.max(1, predictions.length),
    answered: answered.length,
    total: predictions.length,
    selective_accuracy: answered.filter((item) => item.predictedIndex === item.labelIndex).length / Math.max(1, answered.length),
  };
}

async function downloadImage(row) {
  await fs.mkdir(CACHE_DIR, { recursive: true });
  const extension = row.mime === 'image/png' ? '.png' : row.mime === 'image/webp' ? '.webp' : '.jpg';
  const name = crypto.createHash('sha256').update(row.sha1 || row.image_url).digest('hex') + extension;
  const target = path.join(CACHE_DIR, name);
  try { await fs.access(target); return target; } catch {}
  const response = await fetch(row.image_url, { headers: { 'User-Agent': USER_AGENT } });
  if (!response.ok) throw new Error(`Image download failed (${response.status}): ${row.source_url}`);
  await fs.writeFile(target, Buffer.from(await response.arrayBuffer()));
  return target;
}

async function extractEmbeddings(rows) {
  console.log(`Loading ${BASE_MODEL}`);
  const processor = await AutoProcessor.from_pretrained(BASE_MODEL);
  const visionModel = await CLIPVisionModelWithProjection.from_pretrained(BASE_MODEL, { dtype: 'q8' });
  const samples = [];
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    try {
      const localPath = await downloadImage(row);
      const image = await RawImage.read(localPath);
      const inputs = await processor(image);
      const output = await visionModel(inputs);
      const embedding = l2Normalize(Array.from(output.image_embeds.data));
      samples.push({ row, embedding, labelIndex: LABELS.indexOf(row.label) });
      console.log(`[${i + 1}/${rows.length}] ${row.split}/${row.family}/${row.file_title}`);
    } catch (error) {
      console.warn(`Skipping ${row.file_title}: ${error.message}`);
    }
  }
  return samples;
}

function matrixSvg(matrix) {
  const cell = 92, left = 185, top = 90;
  const max = Math.max(1, ...matrix.flat());
  const width = left + cell * LABELS.length + 30;
  const height = top + cell * LABELS.length + 80;
  const rects = [];
  for (let r = 0; r < LABELS.length; r += 1) {
    for (let c = 0; c < LABELS.length; c += 1) {
      const value = matrix[r][c];
      const alpha = 0.08 + 0.82 * value / max;
      rects.push(`<rect x="${left + c * cell}" y="${top + r * cell}" width="${cell - 4}" height="${cell - 4}" rx="8" fill="rgba(32,93,117,${alpha.toFixed(3)})"/><text x="${left + c * cell + (cell - 4) / 2}" y="${top + r * cell + 52}" text-anchor="middle" font-size="22" font-weight="700" fill="${alpha > 0.55 ? '#fff' : '#102638'}">${value}</text>`);
    }
  }
  const columns = LABELS.map((label, index) => `<text transform="translate(${left + index * cell + 42},${top - 12}) rotate(-28)" text-anchor="start" font-size="12" fill="#52616d">${LABEL_NAMES[label]}</text>`).join('');
  const rows = LABELS.map((label, index) => `<text x="${left - 15}" y="${top + index * cell + 48}" text-anchor="end" font-size="12" fill="#52616d">${LABEL_NAMES[label]}</text>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="#fffefa"/><text x="24" y="34" font-family="Georgia,serif" font-size="24" font-weight="600" fill="#102638">Held-out family confusion matrix</text><text x="${left + cell * 2}" y="${height - 18}" text-anchor="middle" font-size="13" fill="#52616d">Predicted class</text><text transform="translate(22,${top + cell * 2}) rotate(-90)" text-anchor="middle" font-size="13" fill="#52616d">True class</text>${columns}${rows}${rects.join('')}</svg>`;
}

function modelCard(metrics, counts, thresholds) {
  const rows = LABELS.map((label) => {
    const item = metrics.test.standard.per_class[label];
    return `| ${LABEL_NAMES[label]} | ${item.support} | ${item.precision.toFixed(3)} | ${item.recall.toFixed(3)} | ${item.f1.toFixed(3)} |`;
  }).join('\n');
  return `# Device Architecture Classifier - Model Card\n\n## Model\n\n- Version: \`${metrics.model_version}\`\n- Base encoder: \`${BASE_MODEL}\`\n- Task-specific head: four-class multinomial logistic regression\n- Execution: fully in-browser through Transformers.js and ONNX Runtime\n- Training compute: standard GitHub Actions CPU runner\n\n## Data\n\nThis pilot uses ${counts.total} Wikimedia Commons images selected from architecture-associated device-family searches. Source URLs, license metadata, labels, families, and fixed train/validation/test assignments are recorded in \`data/manifest.csv\`. The labels are weak labels inferred from the search family and therefore require human review before the results can be treated as a validated study.\n\nThe test set contains device families not used for training. This is stricter than a random image split, although it does not eliminate every visual shortcut.\n\n- Training images: ${counts.train}\n- Validation images: ${counts.validation}\n- Test images: ${counts.test}\n\n## Held-out test results\n\n- Accuracy: **${(metrics.test.standard.accuracy * 100).toFixed(1)}%**\n- Macro F1: **${metrics.test.standard.macro_f1.toFixed(3)}**\n- Selective coverage: **${(metrics.test.selective.coverage * 100).toFixed(1)}%**\n- Accuracy among answered examples: **${(metrics.test.selective.selective_accuracy * 100).toFixed(1)}%**\n\n| Class | Support | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|\n${rows}\n\n## Abstention rule\n\nThe site returns **Indeterminate** when the maximum class probability is below ${thresholds.min_probability.toFixed(3)} or the top-two probability margin is below ${thresholds.min_margin.toFixed(3)}. These thresholds were selected only on the validation split with a minimum 60% validation coverage requirement.\n\n## Limitations\n\n- The training labels are weakly supervised and not yet individually reviewed.\n- Search-derived families can create object-category shortcuts.\n- Exterior photographs cannot prove hidden circuitry, firmware, or internal signal paths.\n- A small pilot dataset is not sufficient for strong reliability claims.\n- Wikimedia search results can contain atypical examples even after automatic filtering.\n\n## Intended use\n\nThis is an experimental educational and research prototype. It must not be used for repair, safety, compliance, purchasing, or engineering decisions.\n`;
}

async function main() {
  const rows = parseCsv(await fs.readFile(MANIFEST_PATH, 'utf8')).filter((row) => LABELS.includes(row.label));
  const samples = await extractEmbeddings(rows);
  const bySplit = Object.groupBy(samples, (sample) => sample.row.split);
  const train = bySplit.train ?? [], validation = bySplit.validation ?? [], test = bySplit.test ?? [];
  if (train.length < 40 || validation.length < 12 || test.length < 12) throw new Error(`Insufficient usable images after extraction: train=${train.length}, validation=${validation.length}, test=${test.length}`);

  const dimensions = train[0].embedding.length;
  const trained = trainLogisticRegression(train, dimensions);
  const validationEval = confusionMatrix(validation, trained.weights, trained.bias);
  const thresholds = tuneThresholds(validationEval.predictions);
  const testEval = confusionMatrix(test, trained.weights, trained.bias);
  const modelVersion = `clip-linear-wikimedia-v0.1-${new Date().toISOString().slice(0, 10)}`;
  const counts = { total: samples.length, train: train.length, validation: validation.length, test: test.length };
  const metrics = {
    model_version: modelVersion,
    generated_at: new Date().toISOString(),
    base_model: BASE_MODEL,
    dataset_version: rows[0]?.dataset_version,
    counts,
    validation: {
      standard: classificationMetrics(validationEval.predictions),
      selective: selectiveMetrics(validationEval.predictions, thresholds),
    },
    test: {
      standard: classificationMetrics(testEval.predictions),
      selective: selectiveMetrics(testEval.predictions, thresholds),
      confusion_matrix: testEval.matrix,
    },
    thresholds,
    hyperparameters: trained.hyperparameters,
  };

  const artifact = {
    schema_version: 1,
    model_version: modelVersion,
    generated_at: metrics.generated_at,
    base_model: BASE_MODEL,
    dataset_version: rows[0]?.dataset_version,
    embedding_dimensions: dimensions,
    labels: LABELS,
    label_names: LABEL_NAMES,
    normalization: 'l2',
    weights: trained.weights.map((row) => row.map((value) => Number(value.toFixed(8)))),
    bias: trained.bias.map((value) => Number(value.toFixed(8))),
    thresholds,
    metrics: {
      test_accuracy: metrics.test.standard.accuracy,
      test_macro_f1: metrics.test.standard.macro_f1,
      test_coverage: metrics.test.selective.coverage,
      test_selective_accuracy: metrics.test.selective.selective_accuracy,
    },
    warning: 'Public-data pilot with weak labels. See results/MODEL_CARD.md.',
  };

  await fs.mkdir(path.dirname(MODEL_PATH), { recursive: true });
  await fs.mkdir(path.dirname(METRICS_PATH), { recursive: true });
  await fs.writeFile(MODEL_PATH, JSON.stringify(artifact, null, 2) + '\n');
  await fs.writeFile(METRICS_PATH, JSON.stringify(metrics, null, 2) + '\n');
  await fs.writeFile(MATRIX_PATH, matrixSvg(testEval.matrix));
  await fs.writeFile(CARD_PATH, modelCard(metrics, counts, thresholds));

  const predictionHeaders = ['split', 'family', 'file_title', 'source_url', 'true_label', 'predicted_label', 'correct', ...LABELS];
  const predictionRows = [...validationEval.predictions, ...testEval.predictions].map((item) => {
    const values = {
      split: item.row.split,
      family: item.row.family,
      file_title: item.row.file_title,
      source_url: item.row.source_url,
      true_label: LABELS[item.labelIndex],
      predicted_label: LABELS[item.predictedIndex],
      correct: item.predictedIndex === item.labelIndex,
      ...Object.fromEntries(LABELS.map((label, index) => [label, item.scores[index].toFixed(8)])),
    };
    return predictionHeaders.map((header) => csvEscape(values[header])).join(',');
  });
  await fs.writeFile(PREDICTIONS_PATH, [predictionHeaders.join(','), ...predictionRows].join('\n') + '\n');

  console.log(JSON.stringify(metrics, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
