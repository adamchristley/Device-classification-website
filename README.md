# Device Architecture AI

A Vercel-hosted research website and browser-based image-classification prototype.

## Live model: zero-shot ensemble v0.2

The current demo uses a quantized CLIP ViT-B/32 visual-language model through Transformers.js. It runs inside a Web Worker in the visitor's browser, so the uploaded image is not sent to a project inference server.

Instead of comparing an image against one sentence per class, version 0.2 uses four prompts for each architecture category and aggregates their scores:

- Analog / Mechanical
- Digital / Electronic
- Software Controlled
- Hybrid

A transparent top-score and score-margin rule returns **Indeterminate** when the four class scores are not clearly separated. These are relative similarity scores, not calibrated probabilities.

## What this model is and is not

This is a genuine working computer-vision baseline, but it is not yet a custom supervised model trained on the project dataset. The next research stage is to collect verified device images, create model-level train, validation, and test splits, and train a lightweight classifier on frozen CLIP embeddings. That model can then be compared directly against this zero-shot baseline.

The current model cannot prove hidden circuitry, firmware, processor architecture, or internal signal paths. It should not be used for repair, safety, compliance, or engineering decisions.

## Run locally

A local web server is required because inference runs in an ES-module Web Worker.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Deployment

The repository is connected to Vercel. Commits pushed to `main` automatically deploy to the production project.

## Project structure

- `index.html`: research landing page and prototype interface
- `styles.css`: responsive research-paper-inspired design
- `app.js`: upload handling, uncertainty rule, and results UI
- `model-worker.js`: in-browser Transformers.js / CLIP inference and prompt aggregation
- `vercel.json`: static deployment and security headers
- `favicon.svg`: site icon
