"""
Dataset de PyTorch para NIH ChestX-ray14.

Decisiones (justificadas en docs/01_fase_analisis.md):

1. LAZY LOADING: las 112,120 imagenes pesan ~42 GB descomprimidas. No se
   precargan en RAM. Cada __getitem__ abre el archivo PNG bajo demanda.

2. MULTI-ETIQUETA: cada item devuelve un tensor de 14 floats (0.0 / 1.0)
   en lugar de un entero de clase. Esto se alinea con la salida sigmoid
   del modelo y la perdida BCE.

3. ROBUSTEZ: si una imagen falta o no se puede leer, se loguea el error
   y se devuelve una imagen negra. Esto evita que el entrenamiento se
   detenga por una imagen corrupta.

Uso esperado:
    from torch.utils.data import DataLoader
    from src.data.dataset import ChestXrayDataset, PATHOLOGIES
    from src.data.transforms import get_transforms

    train_ds = ChestXrayDataset(
        metadata_csv='data/processed/metadata_with_splits.csv',
        images_dir='data/raw/images',
        split='train',
        transform=get_transforms('train'),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
N_CLASSES = len(PATHOLOGIES)


class ChestXrayDataset(Dataset):
    """
    Dataset multi-etiqueta para NIH ChestX-ray14.

    Args:
        metadata_csv: Ruta al CSV con columna 'split' ('train'|'val'|'test')
                      generado por build_splits.py.
        images_dir:   Carpeta donde estan TODOS los PNG (las 112,120 imagenes
                      en una sola carpeta plana, como las extrae el ZIP del NIH).
        split:        Cual de los splits cargar ('train' | 'val' | 'test').
        transform:    Callable que recibe un PIL.Image y devuelve un tensor.
                      Si es None, se devuelve la imagen RGB sin procesar.
        return_meta:  Si True, ademas de (image, label) devuelve un dict con
                      patient_id, age, gender, view, image_id (util para
                      analisis de errores y reportes estratificados).
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        images_dir: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.return_meta = return_meta
        self.split = split

        df = pd.read_csv(metadata_csv)
        if "split" not in df.columns:
            raise ValueError(
                "El CSV no contiene la columna 'split'. "
                "Ejecuta primero: python -m src.data.build_splits"
            )

        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No hay filas para split='{split}' en {metadata_csv}")

        missing_cols = [c for c in PATHOLOGIES if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas de patologias: {missing_cols}")

        self.df = df
        self.image_ids = df["image_id"].astype(str).tolist()
        self.labels = df[PATHOLOGIES].astype("float32").to_numpy()

        logger.info(
            "ChestXrayDataset[%s]: %d imagenes, %d pacientes",
            split, len(df), df["patient_id"].nunique(),
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.images_dir / image_id

        try:
            image = Image.open(image_path).convert("L")
        except (FileNotFoundError, OSError) as exc:
            logger.warning("No se pudo abrir %s (%s). Devolviendo imagen negra.",
                           image_path, exc)
            image = Image.new("L", (224, 224), color=0)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.from_numpy(self.labels[idx])

        if self.return_meta:
            row = self.df.iloc[idx]
            meta = {
                "image_id": image_id,
                "patient_id": int(row["patient_id"]),
                "age": float(row["age"]) if pd.notna(row.get("age")) else -1.0,
                "gender": str(row.get("gender", "")),
                "view": str(row.get("view", "")),
            }
            return image, label, meta

        return image, label


def compute_pos_weight(metadata_csv: str | Path, split: str = "train") -> torch.Tensor:
    """
    Calcula pos_weight para BCEWithLogitsLoss.

    Formula:    pos_weight_i = (#negativos_i) / (#positivos_i)

    Si hay 14 positivos por cada 100 negativos para 'Effusion', su pos_weight
    es 100/14 ~= 7.14, lo que penaliza mas los falsos negativos en esa clase.

    Returns:
        Tensor de shape (14,) con pos_weight por patologia.
    """
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"split '{split}' vacio en {metadata_csv}")

    weights = []
    for patho in PATHOLOGIES:
        pos = float(df[patho].sum())
        neg = float(len(df) - pos)
        weight = neg / pos if pos > 0 else 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
