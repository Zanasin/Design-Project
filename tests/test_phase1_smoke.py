from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog, local_binary_pattern
from torch.optim import AdamW
from torchvision.models import ResNet18_Weights, resnet18
from xgboost import XGBClassifier


def test_cuda_compute_and_mixed_precision() -> None:
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA."

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)

    print("\nCUDA device:", properties.name)
    print("Compute capability:", torch.cuda.get_device_capability(device))
    print("Total VRAM:", f"{properties.total_memory / 1024**3:.2f} GiB")
    print("PyTorch CUDA runtime:", torch.version.cuda)

    torch.manual_seed(42)

    matrix_a = torch.randn(1024, 1024, device=device)
    matrix_b = torch.randn(1024, 1024, device=device)

    result_fp32 = matrix_a @ matrix_b
    torch.cuda.synchronize()

    assert result_fp32.shape == (1024, 1024)
    assert torch.isfinite(result_fp32).all()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result_amp = matrix_a @ matrix_b

    torch.cuda.synchronize()

    assert result_amp.shape == (1024, 1024)
    assert torch.isfinite(result_amp).all()

    print("FP32 matrix output dtype:", result_fp32.dtype)
    print("AMP matrix output dtype:", result_amp.dtype)
    print("CUDA and mixed-precision computation passed.")


def test_pretrained_resnet18_training_step() -> None:
    assert torch.cuda.is_available(), "CUDA is required for this test."

    device = torch.device("cuda")
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 1)
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.fc.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    images = torch.randn(4, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32, device=device)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert logits.shape == (4,)
    assert torch.isfinite(loss)
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()

    allocated = torch.cuda.max_memory_allocated() / 1024**2

    print("\nResNet-18 output shape:", tuple(logits.shape))
    print("Training-step loss:", float(loss.detach().cpu()))
    print("Peak allocated GPU memory:", f"{allocated:.2f} MiB")
    print("Pretrained ResNet-18 training step passed.")


def test_classical_models() -> None:
    features, labels = make_classification(
        n_samples=800,
        n_features=40,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        class_sep=1.5,
        flip_y=0.01,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        stratify=labels,
        random_state=42,
    )

    svm_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    dual="auto",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )

    svm_pipeline.fit(x_train, y_train)
    svm_predictions = svm_pipeline.predict(x_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    xgb_model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        n_jobs=4,
        random_state=42,
    )

    xgb_model.fit(x_train, y_train)
    xgb_predictions = xgb_model.predict(x_test)
    xgb_probabilities = xgb_model.predict_proba(x_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    assert svm_accuracy >= 0.70
    assert xgb_accuracy >= 0.70
    assert xgb_probabilities.shape == (len(x_test), 2)
    assert np.isfinite(xgb_probabilities).all()

    print("\nLinear SVM smoke-test accuracy:", f"{svm_accuracy:.4f}")
    print("XGBoost smoke-test accuracy:", f"{xgb_accuracy:.4f}")
    print("Classical model tests passed.")


def test_image_processing_and_features() -> None:
    image_size = 224

    x_gradient = np.tile(
        np.linspace(0, 255, image_size, dtype=np.uint8),
        (image_size, 1),
    )
    y_gradient = x_gradient.T

    rgb_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = x_gradient
    rgb_image[:, :, 1] = y_gradient
    rgb_image[:, :, 2] = 128

    cv2.circle(
        rgb_image,
        center=(112, 112),
        radius=45,
        color=(255, 255, 255),
        thickness=-1,
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        image_path = Path(temporary_directory) / "smoke_test.png"

        Image.fromarray(rgb_image).save(image_path)

        with Image.open(image_path) as pil_image:
            pil_rgb = pil_image.convert("RGB")
            perceptual_hash = imagehash.phash(pil_rgb)

        cv_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert cv_image is not None

        grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        hog_features = hog(
            grayscale,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        lbp_image = local_binary_pattern(
            grayscale,
            P=8,
            R=1,
            method="uniform",
        )

        fft_spectrum = np.abs(
            np.fft.fftshift(
                np.fft.fft2(grayscale.astype(np.float32))
            )
        )

        dct_coefficients = cv2.dct(
            grayscale.astype(np.float32) / 255.0
        )

        assert pil_rgb.size == (224, 224)
        assert grayscale.shape == (224, 224)
        assert hog_features.ndim == 1
        assert hog_features.size > 0
        assert np.isfinite(hog_features).all()
        assert np.isfinite(lbp_image).all()
        assert np.isfinite(fft_spectrum).all()
        assert np.isfinite(dct_coefficients).all()
        assert len(str(perceptual_hash)) > 0

        print("\nImage shape:", rgb_image.shape)
        print("HOG feature length:", hog_features.size)
        print("LBP shape:", lbp_image.shape)
        print("FFT shape:", fft_spectrum.shape)
        print("DCT shape:", dct_coefficients.shape)
        print("Perceptual hash:", perceptual_hash)
        print("Image-processing and feature tests passed.")
