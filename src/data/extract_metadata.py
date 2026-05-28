"""
Extrae únicamente los archivos de metadatos (CSV y listas de splits) del ZIP
del dataset NIH ChestX-ray14, sin descomprimir las ~45 GB de imágenes.

Uso:
    python src/data/extract_metadata.py --zip "C:/ruta/a/archive.zip"
    python src/data/extract_metadata.py --zip "C:/ruta/a/archive.zip" --out data/raw

El script busca dentro del ZIP los archivos:
  - Data_Entry_2017.csv          (etiquetas de las 112.120 imágenes)
  - BBox_List_2017.csv           (bounding boxes de un subconjunto)
  - train_val_list.txt           (split oficial NIH para train+val)
  - test_list.txt                (split oficial NIH para test)
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

METADATA_FILES = {
    "Data_Entry_2017.csv",
    "Data_Entry_2017_v2020.csv",
    "BBox_List_2017.csv",
    "train_val_list.txt",
    "test_list.txt",
}


def extract_metadata(zip_path: Path, out_dir: Path) -> list[Path]:
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el ZIP: {zip_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        targets = [n for n in names if Path(n).name in METADATA_FILES]

        if not targets:
            print("[!] No se encontró ningún archivo de metadatos en el ZIP.")
            print("    Archivos en el ZIP (primeros 20):")
            for n in names[:20]:
                print(f"      - {n}")
            return extracted

        print(f"[+] Encontrados {len(targets)} archivo(s) de metadatos:")
        for member in targets:
            target_path = out_dir / Path(member).name
            print(f"    -> {target_path}")
            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            extracted.append(target_path)

    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True, type=Path, help="Ruta al ZIP descargado de Kaggle")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
        help="Carpeta destino (por defecto: data/raw)",
    )
    args = parser.parse_args()

    extracted = extract_metadata(args.zip, args.out)
    if not extracted:
        return 1

    print("\n[OK] Listo. Archivos extraídos:")
    for p in extracted:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p}  ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
