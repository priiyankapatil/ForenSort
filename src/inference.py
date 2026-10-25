"""Inference helpers shared by the local review app."""
from pathlib import Path

import cv2
import numpy as np
import torch

from model import QualityNet

GRAY_MEAN, GRAY_STD = 0.449, 0.226
NUM_CLASSES = 6


def prepare_image(image: np.ndarray, size: int = 512) -> tuple[torch.Tensor, np.ndarray, tuple[int, int, int, int]]:
    """Letterbox a grayscale image using the same geometry as training."""
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")
    h, w = image.shape
    if h < 1 or w < 1:
        raise ValueError("Image is empty")
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    x = (canvas.astype(np.float32) / 255.0 - GRAY_MEAN) / GRAY_STD
    tensor = torch.from_numpy(x)[None, None]
    return tensor, canvas, (x0, y0, nw, nh)


def load_model(checkpoint_path: str | Path, device: torch.device) -> QualityNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = QualityNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.inference_mode()
def predict(model: QualityNet, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    prediction, _, _ = predict_with_confidence(model, tensor, device)
    return prediction


@torch.inference_mode()
def predict_with_confidence(
    model: QualityNet, tensor: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted levels, max-class confidence, and normalized entropy."""
    logits = model(tensor.to(device))
    probabilities = logits.softmax(1)
    prediction = probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
    confidence = probabilities.max(1)[0][0].cpu().numpy().astype(np.float32)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)[0]
    entropy = (entropy / np.log(NUM_CLASSES)).cpu().numpy().astype(np.float32)
    return prediction, confidence, entropy


def summarize_prediction(
    prediction: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    content: np.ndarray,
) -> dict[str, float | int | str]:
    """Aggregate regional predictions into an advisory image-level ranking."""
    levels = prediction[content]
    confidences = confidence[content]
    entropies = entropy[content]
    high = (levels >= 4).mean() * 100
    usable = (levels >= 2).mean() * 100
    average = float(levels.mean())
    # The score emphasizes average quality while rewarding broad, usable coverage.
    score = 0.7 * (average / 5 * 100) + 0.3 * usable
    category = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
    return {
        "average_quality": average,
        "high_quality_regions": float(high),
        "usable_regions": float(usable),
        "image_score": float(score),
        "category": category,
        "confidence": float(confidences.mean() * 100),
        "uncertainty": float(entropies.mean()),
    }
