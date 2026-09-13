# Real or Rendered?

Robust detection of AI-generated images across unseen generators and controlled
image degradation.

## Project Task

This project performs binary image classification:

- `0` — real image
- `1` — AI-generated image

The main goal is not only high clean-set accuracy. The project evaluates how
well detectors generalise to generators not seen during training and how much
their performance drops after image degradation.

## Core Models

1. Linear SVM using handcrafted forensic features
2. XGBoost using compact handcrafted forensic features
3. ImageNet-pretrained ResNet-18 with fine-tuning

Swin Transformer Tiny is optional and will only be considered after the
complete three-model project is finished.

## Dataset Plan

The project will use exactly 7,000 balanced images:

- 3,500 real images
- 3,500 AI-generated images

### Training set

- 2,000 real
- 1,000 Stable Diffusion
- 1,000 BigGAN

### Validation set

- 500 real
- 250 Stable Diffusion
- 250 BigGAN

### In-domain test set

- 500 real
- 250 Stable Diffusion
- 250 BigGAN

### Unseen-generator test set

- 500 real
- 250 Midjourney
- 250 GLIDE

Midjourney and GLIDE must not appear in training or validation.

## Handcrafted Feature Families

### Spatial features

- HOG
- edge statistics
- colour statistics
- local contrast

### Texture features

- Local Binary Patterns
- local entropy
- gradient statistics

### Frequency features

- FFT radial energy
- spectral entropy
- DCT band energy
- high-frequency ratios

## Main Evaluations

- In-domain clean classification
- Unseen-generator classification
- JPEG quality 90, 70 and 50
- Resize degradation
- Gaussian blur
- Classical feature ablation
- Generator-wise evaluation
- Failure analysis
- Training and inference efficiency

## Environment Setup

Fedora requires Python 3.12 for this project environment.

```bash
sudo dnf install python3.12 python3.12-devel
./scripts/setup_environment.sh
source .venv/bin/activate
```

PyTorch is installed separately from the CUDA 13.0 wheel index specified in
`requirements-torch.txt`.

## Run Phase 1 Tests

Activate the environment:

```bash
source .venv/bin/activate
```

Run the smoke tests:

```bash
pytest -q -s tests/test_phase1_smoke.py
```

Expected result:

```text
4 passed
```

## Capture Environment Information

```bash
./scripts/capture_environment.sh
cat environment_info.txt
```

## Launch JupyterLab

```bash
jupyter lab
```

Select this kernel:

```text
Python (Real or Rendered)
```

## Project Structure

```text
configs/       Project and experiment configurations
data/          Raw, processed, manifest and degraded data
docs/          Project charter and validation records
features/      Cached handcrafted features
models/        Saved SVM, XGBoost and ResNet-18 models
notebooks/     Exploratory analysis notebooks
predictions/   Per-image prediction files
results/       Metrics, plots and failure cases
scripts/       Reproducible command-line workflows
src/           Reusable project source code
tests/         Automated validation and smoke tests
report/        Final report figures and tables
```

## Project Status

- Phase 1: environment and reproducibility
- Phase 2: small end-to-end prototype
- Phase 3: final 7,000-image dataset
- Phase 4: handcrafted feature extraction
- Phase 5: Linear SVM
- Phase 6: XGBoost
- Phase 7: ResNet-18 fine-tuning
- Phase 8: degradation pipeline
- Phase 9: unified evaluation
- Phase 10: failure analysis
- Phase 11: report, presentation and viva

## Reproducibility Rules

- Use fixed random seeds.
- Deduplicate images before splitting.
- Keep Midjourney and GLIDE out of training and validation.
- Fit scalers, PCA and preprocessing only on training data.
- Do not tune models using either final test set.
- Apply degradations equally to real and AI-generated images.
- Save manifests, configurations, models and per-image predictions.
