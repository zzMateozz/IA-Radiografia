"""Grad-CAM sobre DenseNet-121 (denseblock4)."""

from __future__ import annotations

import io

import matplotlib.cm as mpl_cm
import numpy as np
import torch
from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def generate_gradcam(model, img_tensor: torch.Tensor, target_idx: int) -> np.ndarray:
    target_layer = model.backbone.features.denseblock4
    gradients, activations = [], []

    def save_grad(grad):
        gradients.append(grad)

    def fwd_hook(_module, _inp, out):
        activations.append(out)
        out.register_hook(save_grad)

    hook = target_layer.register_forward_hook(fwd_hook)
    inp = img_tensor.unsqueeze(0).requires_grad_(True)
    logits = model(inp)
    model.zero_grad()
    logits[0, target_idx].backward()
    hook.remove()

    grads = gradients[0].squeeze().detach().numpy()
    acts = activations[0].squeeze().detach().numpy()
    weights = grads.mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * acts).sum(axis=0), 0)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def make_gradcam_overlay(model, img_tensor, target_idx):
    cam = generate_gradcam(model, img_tensor, target_idx)
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224)))
    img_rgb = denormalize(img_tensor)
    heatmap = mpl_cm.jet(cam_resized)[:, :, :3]
    overlay = np.clip(0.55 * img_rgb + 0.45 * heatmap, 0, 1)
    return img_rgb, cam_resized, overlay


def ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()
