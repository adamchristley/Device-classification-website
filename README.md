# Device Architecture AI

A Vercel-ready research website and browser-based zero-shot image-classification prototype.

## What the prototype does

The site lets a user upload one clearly framed image of a device. A quantized CLIP model runs inside a Web Worker in the browser and compares the image against four descriptive architecture classes:

- Analog / Mechanical
- Digital / Electronic
- Software Controlled
- Hybrid

A transparent score-and-margin rule returns **Indeterminate** when the visual evidence is not clearly separated. The scores are labeled as relative similarity scores, not calibrated probabilities.

## Run locally

A local web server is required because the model runs in an ES-module Web Worker.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Deploy to Vercel

1. Push these files to the root of `adamchristley/Device-classification-website`.
2. In Vercel, choose **Add New → Project**.
3. Import the GitHub repository.
4. Keep Framework Preset as **Other**.
5. Leave Build Command and Output Directory blank.
6. Deploy.

Vercel will assign a stable production URL. Future pushes to `main` update the same site.

## Project structure

- `index.html`: research landing page and prototype interface
- `styles.css`: responsive research-paper-inspired visual design
- `app.js`: upload handling, uncertainty rule, and results UI
- `model-worker.js`: in-browser Transformers.js / CLIP inference
- `vercel.json`: static deployment and security headers
- `favicon.svg`: site icon

## Research limitations

This is a zero-shot baseline. It compares visual similarity and cannot prove hidden architecture, inspect circuitry, or verify the presence of firmware. It should not be used for repair, safety, compliance, or engineering decisions.
