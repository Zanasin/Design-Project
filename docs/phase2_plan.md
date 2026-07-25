# Phase 2 Plan — Small End-to-End Prototype

## Purpose

Phase 2 validates the complete project pipeline on a small dataset before the
final 7,000-image dataset is constructed.

The prototype is for engineering validation only. Its model scores are not
final research results.

## Prototype Dataset

The prototype will use 1,000 balanced CIFAKE images:

- 500 real
- 500 AI-generated

The split is:

- Training: 700 images
- Validation: 150 images
- Test: 150 images

Every split remains class-balanced.

## Required Pipeline

1. Acquire the prototype images.
2. Assign stable image IDs.
3. Build a CSV manifest.
4. Verify paths, labels and class counts.
5. Decode and convert images to RGB.
6. Resize images to 224 by 224.
7. Extract minimal spatial, texture and frequency features.
8. Train Linear SVM.
9. Train XGBoost.
10. Fine-tune the pretrained ResNet-18 classification head.
11. Save per-image predictions.
12. Calculate common evaluation metrics.
13. Generate confusion matrices.
14. Record runtime and detected errors.

## Prototype Models

### Linear SVM

Uses scaled handcrafted features.

### XGBoost

Uses the same compact handcrafted feature set.

### ResNet-18

Uses ImageNet-pretrained weights. The backbone remains frozen during the
prototype and only the final classification layer is trained.

## Restrictions

- Do not interpret prototype scores as final research findings.
- Do not use the prototype test split for model selection.
- Do not copy prototype images into the final dataset without a new audit.
- Do not begin full hyperparameter tuning.
- Do not implement degradation experiments in this phase.
- Do not implement Swin-T in this phase.

## Definition of Done

Phase 2 is complete when all three models:

- load the same prototype split;
- train without pipeline errors;
- produce probability or decision outputs;
- produce test predictions;
- use the common evaluation system;
- save reproducible artifacts;
- pass Phase 2 validation tests.
