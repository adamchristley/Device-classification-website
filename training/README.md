# Free training and evaluation pipeline

This directory contains the reproducible training system for the device-architecture classifier. It is designed to run without paid services.

## Architecture

1. `collect-dataset.mjs` queries the Wikimedia Commons API using the family definitions in `data/dataset_config.json`.
2. It filters non-photographic and low-resolution files, deduplicates by Commons SHA-1, and records source and license metadata in `data/manifest.csv`.
3. `train-model.mjs` downloads thumbnails only for the duration of the run.
4. The same quantized CLIP ViT-B/32 vision encoder used by the website extracts a normalized 512-dimensional embedding for every image.
5. A four-class multinomial logistic-regression head is trained on the frozen embeddings.
6. Validation data selects the probability and top-two-margin thresholds used for `Indeterminate`.
7. Entire device families are held out for validation and testing.
8. The workflow exports browser-ready weights, metrics, predictions, a confusion matrix, and a model card.

## Cost

The repository is public, so the workflow uses a standard GitHub-hosted runner without billable Actions minutes. The website remains a static Vercel Hobby deployment, and inference runs in the visitor's browser. No GPU service, database, API key, or inference server is required.

## Reproduce locally

Requirements: Node.js 22 and npm.

```bash
cd training
npm install --no-audit --no-fund
npm run all
```

Generated files:

- `data/manifest.csv`
- `models/classifier.json`
- `results/metrics.json`
- `results/predictions.csv`
- `results/confusion-matrix.svg`
- `results/MODEL_CARD.md`

## Scientific status

The automated dataset is a pilot with weak labels derived from device-family searches. It makes the project measurable and reproducible, but it is not a substitute for human verification. Every manifest row has `review_status=weak-label-needs-review`. The model card reports this limitation explicitly.
