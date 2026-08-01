# Device Architecture Classifier - Model Card

## Model

- Version: `clip-linear-wikimedia-v0.1-2026-08-01`
- Base encoder: `Xenova/clip-vit-base-patch32`
- Task-specific head: four-class multinomial logistic regression
- Execution: fully in-browser through Transformers.js and ONNX Runtime
- Training compute: standard GitHub Actions CPU runner

## Data

This pilot uses 189 Wikimedia Commons images selected from architecture-associated device-family searches. Source URLs, license metadata, labels, families, and fixed train/validation/test assignments are recorded in `data/manifest.csv`. The labels are weak labels inferred from the search family and therefore require human review before the results can be treated as a validated study.

The test set contains device families not used for training. This is stricter than a random image split, although it does not eliminate every visual shortcut.

- Training images: 127
- Validation images: 32
- Test images: 30

## Held-out test results

- Accuracy: **26.7%**
- Macro F1: **0.279**
- Selective coverage: **50.0%**
- Accuracy among answered examples: **33.3%**

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Analog / Mechanical | 8 | 0.571 | 0.500 | 0.533 |
| Digital / Electronic | 8 | 0.077 | 0.125 | 0.095 |
| Software Controlled | 8 | 0.500 | 0.125 | 0.200 |
| Hybrid | 6 | 0.250 | 0.333 | 0.286 |

## Abstention rule

The site returns **Indeterminate** when the maximum class probability is below 0.250 or the top-two probability margin is below 0.225. These thresholds were selected only on the validation split with a minimum 60% validation coverage requirement.

## Limitations

- The training labels are weakly supervised and not yet individually reviewed.
- Search-derived families can create object-category shortcuts.
- Exterior photographs cannot prove hidden circuitry, firmware, or internal signal paths.
- A small pilot dataset is not sufficient for strong reliability claims.
- Wikimedia search results can contain atypical examples even after automatic filtering.

## Intended use

This is an experimental educational and research prototype. It must not be used for repair, safety, compliance, purchasing, or engineering decisions.
