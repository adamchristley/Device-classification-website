# Independent attribute model design

## Motivation

The prior four-way softmax-style presentation made every architecture class receive a portion of a fixed total. A low Mechanical score could therefore look like positive mechanical evidence even when it only represented the least-similar forced option. The architecture categories also overlap: a device can be digital and mechanical, or digital and software controlled.

## Inference stages

### 1. Physical-device gate

The image is compared against positive device descriptions and opposing non-device or unidentifiable-image descriptions. When the positive evidence ratio is below the gate threshold, the system returns `Indeterminate` without deriving an architecture label.

### 2. Independent evidence axes

Each axis has an equal number of positive and opposing prompts:

- Mechanical
- Analog
- Digital
- Software control

For one axis, the reported score is:

```text
sum(positive prompt similarity)
-----------------------------------------------
sum(positive similarity) + sum(opposing similarity)
```

The four axis scores are independent and do not sum to 100%.

### 3. Rule-derived label

The current transparent rules are:

- Physical or analog evidence plus digital or software evidence -> `Hybrid`
- Software evidence without physical or analog evidence -> `Software Controlled`
- Digital evidence without physical or analog evidence -> `Digital / Electronic`
- Mechanical or analog evidence without digital or software evidence -> `Analog / Mechanical`
- Weak, conflicting, or unsupported evidence -> `Indeterminate`

## Current thresholds

- Physical-device gate: `0.56`
- Attribute present: `0.60`
- Minimum architecture evidence: `0.55`
- Moderate evidence label: `0.62`
- High evidence label: `0.72`

These thresholds are provisional. They must be tuned on a locked validation set and then evaluated once on held-out device families and non-device negatives.

## Required next dataset

The next annotation format should be multi-label rather than one mutually exclusive class per image:

```text
device_present
mechanical_present
analog_present
digital_present
software_control_present
final_architecture_label
review_status
label_reason
```

The benchmark should include non-device images, ambiguous crops, packaging, diagrams, people holding devices, and difficult hybrid devices. Attribute metrics and final-label metrics should be reported separately.
