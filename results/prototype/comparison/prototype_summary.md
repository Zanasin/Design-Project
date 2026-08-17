# Prototype Model Comparison

## Scope

All three finalized models were evaluated on the same 150 test images with identical labels: 75 real and 75 AI-generated.

## Test metrics

| Model | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear SVM | 0.7400 | 0.7400 | 0.7500 | 0.7200 | 0.7600 | 0.7347 | 0.8084 |
| XGBoost | 0.8600 | 0.8600 | 0.8857 | 0.8267 | 0.8933 | 0.8552 | 0.9250 |
| ResNet-18 | 0.7400 | 0.7400 | 0.6731 | 0.9333 | 0.5467 | 0.7821 | 0.8414 |

## Confusion matrices

| Model | TN | FP | FN | TP | Correct |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 57 | 18 | 21 | 54 | 111 |
| XGBoost | 67 | 8 | 13 | 62 | 129 |
| ResNet-18 | 41 | 34 | 5 | 70 | 111 |

## Metric leaders

- Accuracy: XGBoost (0.8600)
- Balanced Accuracy: XGBoost (0.8600)
- Precision: XGBoost (0.8857)
- Recall: ResNet-18 (0.9333)
- F1: XGBoost (0.8552)
- Roc Auc: XGBoost (0.9250)

## Model agreement

- All three agree: 83 / 150
- At least one model disagrees: 67 / 150
- All three correct: 77 / 150
- All three wrong: 6 / 150
- Only Linear SVM correct: 4
- Only XGBoost correct: 7
- Only ResNet-18 correct: 3

## Artifact sizes

| Model | Bytes | MiB |
|---|---:|---:|
| Linear SVM | 778,223 | 0.74 |
| XGBoost | 128,031 | 0.12 |
| ResNet-18 | 44,786,827 | 42.71 |

## Interpretation

- XGBoost is the strongest overall prototype according to F1.
- ResNet-18 has the highest AI-image recall.
- XGBoost has the smallest saved model artifact.
- The disagreement results show that the models learn partially different decision patterns.
- Images misclassified by all three models should be prioritized during later failure analysis.

## Limitations

- These are prototype results from 1,000 CIFAKE images, not final research results.
- The prototype does not evaluate generalization to unseen image generators.
- CIFAKE images are low-resolution and may contain dataset-specific artifacts.
- The final conclusions must come from the generator-disjoint 7,000-image dataset.
