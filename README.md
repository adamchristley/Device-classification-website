# Device Architecture AI

A Vercel-hosted research website and browser-based device-architecture evidence prototype.

## Live design: independent attribute baseline v0.3

The demo uses a quantized CLIP ViT-B/32 model through Transformers.js in a Web Worker. Images are processed in the visitor's browser and are not sent to a project inference server.

Version 0.3 replaces the forced four-class ranking with a staged design:

1. **Physical-device gate** - checks whether the image clearly shows a functional device, machine, appliance, instrument, or electronic product.
2. **Independent evidence axes** - scores Mechanical, Analog, Digital, and Software-control evidence separately.
3. **Transparent derivation rules** - converts supported attribute combinations into Analog / Mechanical, Digital / Electronic, Software Controlled, Hybrid, or Indeterminate.
4. **Abstention** - returns Indeterminate when the device gate fails, architecture evidence is weak, or the evidence combination is borderline.

The architecture scores do not compete for a fixed 100% total. Each score is a positive-versus-negative CLIP evidence ratio for that attribute. The values are not calibrated probabilities.

## Why this is better than v0.2

The earlier model normalized four class similarities to 100%. That guaranteed every class received some share, even when an attribute was clearly absent. The new design allows Mechanical evidence to remain low for a smartphone while Digital and Software evidence can both be high. It also allows a stereo or appliance to express both physical and electronic evidence and derive a Hybrid result.

## Current supervised model status

The first frozen-CLIP classifier was trained and evaluated through the free GitHub Actions pipeline, but it did not clear the promotion gate. Its results remain documented under `results/`. The browser demo therefore uses the independent zero-shot evidence baseline while a multi-label attribute dataset is developed.

## Run locally

A local web server is required because inference runs in an ES-module Web Worker.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Deployment

The repository is connected to Vercel. Commits pushed to `main` automatically deploy to the production project.

## Project structure

- `index.html` - research landing page and prototype interface
- `styles.css` - responsive research-paper-inspired design
- `app.js` - upload handling and independent-evidence results UI
- `model-worker.js` - device gate, attribute scoring, and rule-derived classification
- `training/` - reproducible public-data training and evaluation pipeline
- `results/` - metrics, predictions, confusion matrix, and model card
- `vercel.json` - static deployment and security headers
