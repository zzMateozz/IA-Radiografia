"""Carga del modelo y predicción sobre radiografías."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.config import MODELS_DIR, PATHOLOGIES
from src.data.transforms import get_eval_transforms
from src.models.densenet_chexnet import DenseNetCheXNet, N_CLASSES


@st.cache_resource(show_spinner="Cargando modelo DenseNet-121…")
def load_model():
    path = MODELS_DIR / "best_model.pt"
    if not path.exists():
        return None, None
    model = DenseNetCheXNet(n_classes=N_CLASSES, pretrained=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data(show_spinner=False)
def load_test_results():
    path = MODELS_DIR / "test_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict(model, img_pil: Image.Image):
    transform = get_eval_transforms()
    tensor = transform(img_pil)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor.unsqueeze(0))).squeeze().numpy()
    return probs, tensor
