# Project Charter

## Title

Real or Rendered? Robust Detection of AI-Generated Images Across Unseen
Generators and Image Degradation

## Task

Binary image classification:

- Real image
- AI-generated image

## Core Models

1. Linear SVM
2. XGBoost
3. ImageNet-pretrained ResNet-18 with fine-tuning

Swin Transformer Tiny is optional and will only be implemented after all core
experiments are complete.

## Dataset

The project will use exactly 7,000 images:

- 3,500 real
- 3,500 AI-generated

Stable Diffusion and BigGAN will be used for training, validation, and
in-domain testing.

Midjourney and GLIDE will remain unseen during training and validation and
will be reserved for cross-generator testing.

## Main Research Question

How reliably do handcrafted-feature classical models and a fine-tuned CNN
detect AI-generated images when evaluated on familiar generators, unseen
generators, and degraded images?

## Main Experiments

1. In-domain clean evaluation
2. Unseen-generator clean evaluation
3. JPEG compression robustness
4. Resize robustness
5. Blur robustness
6. Classical feature ablation
7. Generator-wise analysis
8. Failure analysis

## Success Criterion

Success is not defined only by maximum clean accuracy. The strongest detector
should also:

- generalise to unseen generators;
- retain performance after image degradation;
- have a small clean-to-degraded F1 drop;
- show consistent performance across generators.

## Scope Control

The final unseen-generator test set must not be used for model selection.

Swin-T, ensembles, Grad-CAM, and additional degradations are optional and must
not delay the three-model project.
