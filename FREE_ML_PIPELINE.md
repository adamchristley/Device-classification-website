# Fully free model lifecycle

This project is intentionally designed so that collection, training, evaluation, hosting, and inference can operate without a paid service.

## Services and responsibilities

| Component | Free service | What happens there |
|---|---|---|
| Source code and model artifacts | Public GitHub repository | Version control, documentation, manifest, exported weights, and evaluation files |
| Dataset discovery | Wikimedia Commons API | Finds reusable device photographs and returns source/license metadata |
| Training and evaluation | GitHub Actions standard Ubuntu runner | Extracts CLIP embeddings, trains the classifier head, evaluates held-out families, and exports artifacts |
| Website hosting | Vercel Hobby static deployment | Serves HTML, CSS, JavaScript, model metadata, and the small trained classifier file |
| Base model delivery | Hugging Face Hub | Delivers the quantized CLIP ONNX encoder used by Transformers.js |
| Inference | Visitor's browser | Processes the selected image locally; the project does not upload it to a model server |

## Automatic lifecycle

1. The training workflow reads `data/dataset_config.json`.
2. It builds `data/manifest.csv` from Wikimedia Commons and preserves attribution/license fields.
3. It extracts frozen CLIP image embeddings.
4. It trains a four-class multinomial logistic-regression head.
5. It selects an abstention rule on the validation families.
6. It evaluates only once on held-out test families.
7. It exports `models/classifier.json` and the complete files under `results/`.
8. GitHub commits those outputs to `main`.
9. Vercel detects the commit and redeploys the same production URL.
10. The browser worker detects the trained artifact and uses it. If the artifact is unavailable, it falls back to the documented zero-shot baseline.

## Cost controls

- The repository must stay public for standard GitHub-hosted Actions minutes to remain free.
- The site remains static, so it does not consume a paid inference server or persistent function.
- The large CLIP encoder stays on the Hugging Face Hub and is cached in each browser.
- Only the small linear-classifier artifact is stored in the GitHub repository and served by Vercel.
- No payment method, GPU rental, database, API key, or proprietary dataset is required by the pipeline.

## What counts as an actual model

The custom artifact is a trained multinomial logistic-regression classifier over frozen 512-dimensional CLIP image embeddings. Its weights are learned from the project dataset. It is therefore task-specific, unlike the initial prompt-only baseline.

The pilot dataset is weakly labeled from search families. The generated model card clearly separates a reproducible pilot model from a human-reviewed research model. Accuracy claims should be upgraded only after the manifest has been reviewed image by image.
