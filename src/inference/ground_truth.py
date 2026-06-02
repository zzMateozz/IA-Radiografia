"""Consulta de etiquetas reales del NIH por nombre de imagen."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import PATHOLOGIES, PROJECT_ROOT

METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_clean.csv"


@lru_cache(maxsize=1)
def _load_metadata() -> pd.DataFrame | None:
    if not METADATA_PATH.exists():
        return None
    cols = ["image_id", "labels_raw", "No Finding", *PATHOLOGIES]
    return pd.read_csv(METADATA_PATH, usecols=cols).set_index("image_id")


def get_ground_truth(filename: str) -> dict | None:
    """Etiquetas NIH para una imagen del dataset (por nombre de archivo)."""
    df = _load_metadata()
    if df is None:
        return None

    image_id = Path(filename).name
    if image_id not in df.index:
        return None

    row = df.loc[image_id]
    positives = [p for p in PATHOLOGIES if int(row[p]) == 1]
    labels = {p: int(row[p]) for p in PATHOLOGIES}

    return {
        "image_id": image_id,
        "labels_raw": str(row["labels_raw"]),
        "no_finding": int(row["No Finding"]) == 1,
        "positives": positives,
        "labels": labels,
    }


def _prob_note(nih_pos: bool, prob: float) -> tuple[str, str]:
    """Nota interpretativa sin umbral binario. Retorna (texto, kind)."""
    if nih_pos:
        if prob >= 0.5:
            return "Probabilidad alta — coherente con NIH", "success"
        if prob >= 0.2:
            return "Probabilidad moderada — NIH la etiqueta", "medium"
        return "Probabilidad baja — NIH la etiqueta", "high"
    if prob >= 0.5:
        return "Probabilidad elevada sin etiqueta NIH", "high"
    if prob >= 0.2:
        return "Probabilidad moderada", "medium"
    return "Probabilidad baja", "neutral"


def summarize_nih_comparison(probs, ground_truth: dict) -> dict:
    """Compara etiquetas NIH con probabilidades del modelo (sin umbral binario)."""
    nih_labels = ground_truth["labels"]
    rows = []

    for i, patho in enumerate(PATHOLOGIES):
        nih_pos = nih_labels[patho] == 1
        prob = float(probs[i])
        note, note_kind = _prob_note(nih_pos, prob)
        rows.append({
            "pathology": patho,
            "nih": nih_pos,
            "prob": prob,
            "note": note,
            "note_kind": note_kind,
        })

    nih_pos_rows = [r for r in rows if r["nih"]]
    high_without_label = [r for r in rows if not r["nih"] and r["prob"] >= 0.5]
    top_probs = sorted(rows, key=lambda r: r["prob"], reverse=True)[:5]

    return {
        "rows": rows,
        "nih_positives": [r["pathology"] for r in nih_pos_rows],
        "top_probs": top_probs,
        "high_without_label": high_without_label,
        "n_nih_labeled_high": sum(1 for r in nih_pos_rows if r["prob"] >= 0.5),
        "n_nih_labeled": len(nih_pos_rows),
    }
