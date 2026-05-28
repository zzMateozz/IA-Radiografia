"""
Construye los splits train / val / test del dataset NIH ChestX-ray14.

Estrategia (justificada en docs/01_fase_analisis.md, secciones 6 y 7):

1. Se respetan los splits OFICIALES del NIH:
       - train_val_list.txt -> 86,524 imágenes de 28,008 pacientes.
       - test_list.txt      -> 25,596 imágenes de  2,797 pacientes.
   El test set se reserva exclusivamente para evaluación final.

2. Sobre el conjunto train_val se aplica GroupKFold con groups=patient_id
   para construir el split de validación interno. Esto evita el data leakage
   por paciente (un paciente puede tener hasta 184 imágenes; si se repartieran
   entre train y val el modelo "memorizaría" al paciente).

Salidas (en data/processed/):
    metadata_with_splits.csv   -> CSV con la columna 'split' ('train' | 'val' | 'test').
    splits_summary.json        -> Resumen estadístico de la división (#imágenes
                                  y #pacientes por split, prevalencia por clase).

Uso:
    python -m src.data.build_splits --fold 0 --n-splits 5 --seed 42

Argumentos:
    --fold        Índice del fold de GroupKFold (0..n_splits-1) que se usa
                  como conjunto de validación. Default: 0.
    --n-splits    Número total de folds (default: 5 -> validación = 20% de train_val).
    --seed        Semilla aleatoria. Default: 42.
    --metadata    Ruta al CSV de metadatos. Default: data/processed/metadata_clean.csv.
    --output      Ruta del CSV de salida. Default: data/processed/metadata_with_splits.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold de validacion (0..n_splits-1). Default: 0")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Numero de folds de GroupKFold. Default: 5")
    parser.add_argument("--seed", type=int, default=42, help="Semilla. Default: 42")
    parser.add_argument("--metadata", type=Path,
                        default=project_root / "data" / "processed" / "metadata_clean.csv")
    parser.add_argument("--output", type=Path,
                        default=project_root / "data" / "processed" / "metadata_with_splits.csv")
    return parser.parse_args()


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el CSV: {path}\n"
            "Ejecuta primero el notebook 01_eda_metadata.ipynb."
        )
    df = pd.read_csv(path)
    required = {"image_id", "patient_id", "split_oficial", *PATHOLOGIES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df


def build_splits(df: pd.DataFrame, fold: int, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Asigna a cada fila el split correspondiente: 'train', 'val' o 'test'.

    Las filas con split_oficial == 'test' van directo a 'test'.
    Las filas con split_oficial == 'train_val' se dividen con GroupKFold
    sobre patient_id, y el fold indicado se usa como 'val'.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""

    is_test = df["split_oficial"] == "test"
    df.loc[is_test, "split"] = "test"

    train_val_df = df.loc[~is_test].reset_index(drop=False)
    if train_val_df.empty:
        raise ValueError("No hay imagenes con split_oficial='train_val'.")

    groups = train_val_df["patient_id"].to_numpy()
    unique_patients = np.unique(groups)
    rng.shuffle(unique_patients)
    patient_to_shuffle_idx = {pid: i for i, pid in enumerate(unique_patients)}
    shuffled_groups = np.array([patient_to_shuffle_idx[pid] for pid in groups])

    gkf = GroupKFold(n_splits=n_splits)

    if not 0 <= fold < n_splits:
        raise ValueError(f"fold debe estar entre 0 y {n_splits - 1}, recibido: {fold}")

    fold_assignments = np.full(len(train_val_df), -1, dtype=int)
    for fold_idx, (_train_idx, val_idx) in enumerate(
        gkf.split(train_val_df, groups=shuffled_groups)
    ):
        fold_assignments[val_idx] = fold_idx

    val_mask = fold_assignments == fold
    val_original_idx = train_val_df.loc[val_mask, "index"].to_numpy()
    train_original_idx = train_val_df.loc[~val_mask, "index"].to_numpy()

    df.loc[train_original_idx, "split"] = "train"
    df.loc[val_original_idx, "split"] = "val"

    assert (df["split"] != "").all(), "Hay filas sin split asignado."
    return df


def summarize_splits(df: pd.DataFrame) -> dict:
    """Genera estadisticas de la division para verificar que es razonable."""
    summary: dict = {"splits": {}}
    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name]
        prevalence = {
            patho: round(float(split_df[patho].mean()) * 100, 2) for patho in PATHOLOGIES
        }
        summary["splits"][split_name] = {
            "n_images": int(len(split_df)),
            "n_patients": int(split_df["patient_id"].nunique()),
            "pct_images": round(len(split_df) / len(df) * 100, 2),
            "prevalence_pct": prevalence,
        }
    summary["leakage_check"] = check_leakage(df)
    return summary


def check_leakage(df: pd.DataFrame) -> dict:
    """Verifica que ningun paciente aparezca en mas de un split."""
    train_p = set(df.loc[df["split"] == "train", "patient_id"])
    val_p = set(df.loc[df["split"] == "val", "patient_id"])
    test_p = set(df.loc[df["split"] == "test", "patient_id"])
    return {
        "train_val_overlap": len(train_p & val_p),
        "train_test_overlap": len(train_p & test_p),
        "val_test_overlap": len(val_p & test_p),
        "ok": len(train_p & val_p) == 0
              and len(train_p & test_p) == 0
              and len(val_p & test_p) == 0,
    }


def main() -> None:
    args = parse_args()

    print(f"[build_splits] Cargando metadatos: {args.metadata}")
    df = load_metadata(args.metadata)
    print(f"[build_splits] Filas: {len(df):,}  Pacientes: {df['patient_id'].nunique():,}")

    print(f"[build_splits] GroupKFold(n_splits={args.n_splits}, fold_val={args.fold}, seed={args.seed})")
    df_split = build_splits(df, fold=args.fold, n_splits=args.n_splits, seed=args.seed)

    summary = summarize_splits(df_split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_csv(args.output, index=False)
    print(f"[build_splits] CSV escrito: {args.output}")

    summary_path = args.output.with_name("splits_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[build_splits] Resumen JSON: {summary_path}")

    print("\n=== RESUMEN ===")
    for split_name, stats in summary["splits"].items():
        print(
            f"  {split_name:>5}: {stats['n_images']:>7,} imgs "
            f"({stats['pct_images']:>5.2f}%)  |  "
            f"{stats['n_patients']:>6,} pacientes"
        )
    leak = summary["leakage_check"]
    flag = "OK" if leak["ok"] else "FAIL"
    print(f"\n  Leakage check: {flag}  "
          f"(train-val={leak['train_val_overlap']}, "
          f"train-test={leak['train_test_overlap']}, "
          f"val-test={leak['val_test_overlap']})")


if __name__ == "__main__":
    main()
