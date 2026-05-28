# Sistema de Clasificación de Radiografías de Tórax (NIH ChestX-ray14)

**Proyecto de Aula — Inteligencia Artificial — III Corte**  
Universidad Popular del Cesar — Facultad de Ingeniería de Sistemas — 2026-I

**Integrantes:**
- Mateo Lopez Patiño
- Anaclaudia Vega Martinez
- Tonny Enrique Jimenez Marquez

---

## ¿Qué hace este proyecto?

Clasifica automáticamente **14 patologías torácicas** a partir de radiografías de tórax usando una red neuronal DenseNet-121 entrenada sobre el dataset público NIH ChestX-ray14 (112.120 imágenes).

---

## Requisitos previos

Antes de clonar, asegúrate de tener instalado:

| Herramienta | Versión mínima | Cómo verificar |
|---|---|---|
| Python | 3.10 o superior | `python --version` |
| Git | cualquiera | `git --version` |
| VSCode | cualquiera | (con extensión **Jupyter** instalada) |

> **Nota GPU:** El entrenamiento del modelo requiere GPU NVIDIA (CUDA). Si solo tienes CPU o GPU AMD, puedes correr el EDA y la inferencia localmente, pero el entrenamiento se hace en **Kaggle** (gratis, con Tesla T4).

---

## Paso a paso: configuración desde cero

### 1. Clonar el repositorio

```bash
git clone https://github.com/zzMateozz/IA-Radiografia.git
cd IA-Radiografia
```

---

### 2. Crear el entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

**Mac / Linux:**
```bash
python3 -m venv .venv
```

> Esto crea una carpeta `.venv/` con Python aislado para el proyecto.

---

### 3. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea scripts, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Mac / Linux:**
```bash
source .venv/bin/activate
```

Cuando está activo verás `(.venv)` al inicio de la terminal.

---

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Esto instala PyTorch, pandas, scikit-learn, matplotlib, jupyter y todo lo necesario.  
> Tarda ~3-5 minutos la primera vez.

---

### 5. Registrar el kernel de Jupyter para VSCode

```bash
python -m ipykernel install --user --name=proyecto_fase3 --display-name="Python (proyecto_fase3)"
```

> Esto hace que VSCode pueda ver el entorno virtual como kernel de notebooks.

---

### 6. Abrir en VSCode

```bash
code .
```

O desde VSCode: **File → Open Folder** → selecciona la carpeta `IA-Radiografia`.

---

### 7. Seleccionar el kernel en el notebook

1. Abre cualquier notebook en `notebooks/`
2. Esquina superior derecha → haz clic en el selector de kernel
3. Busca la sección **Jupyter Kernels**
4. Selecciona **Python (proyecto_fase3)**

> Si no aparece en "Jupyter Kernels", busca en "Python Environments" el que muestre `.venv` en la ruta, o recarga VSCode con `Ctrl+Shift+P` → `Developer: Reload Window`.

---

## Estructura del proyecto

```
IA-Radiografia/
├── data/
│   ├── raw/              # CSV de metadata NIH (Data_Entry_2017.csv, etc.)
│   └── processed/        # Splits generados (metadata_with_splits.csv)
├── notebooks/
│   ├── 01_eda_metadata.ipynb       # Análisis exploratorio completo
│   └── 02_kaggle_training.ipynb    # Entrenamiento (ejecutar en Kaggle)
├── src/
│   ├── data/
│   │   ├── build_splits.py         # Genera los splits train/val/test
│   │   ├── dataset.py              # Dataset de PyTorch
│   │   └── transforms.py           # Augmentación de imágenes
│   └── models/
│       └── densenet_chexnet.py     # Arquitectura DenseNet-121
├── docs/
│   ├── 01_fase_analisis.md         # Documento fase de análisis
│   └── 02_fase_diseno.md           # Documento fase de diseño
├── outputs/
│   ├── figures/                    # Gráficas del EDA
│   └── models/                     # Pesos del modelo (generados al entrenar)
├── md_to_pdf.py                    # Script para convertir .md a PDF
├── requirements.txt
└── README.md
```

---

## Flujo de trabajo del proyecto

```
[1] EDA local          → notebooks/01_eda_metadata.ipynb
[2] Build splits       → python src/data/build_splits.py (ya generado)
[3] Entrenamiento      → subir notebooks/02_kaggle_training.ipynb a Kaggle
[4] Descargar modelo   → copiar best_model.pt a outputs/models/
[5] Evaluación local   → (próximo: 03_evaluacion_resultados.ipynb)
[6] Demo               → (próximo: app Streamlit)
```

---

## Estrategia de cómputo

| Tarea | Entorno | Razón |
|---|---|---|
| EDA y preprocesamiento | PC local (CPU) | Solo metadata, no necesita GPU |
| Entrenamiento del modelo | Kaggle (Tesla T4 gratis) | GPU NVIDIA necesaria (CUDA) |
| Inferencia / demo | PC local (CPU) | Modelo ya entrenado |

---

## Convertir documentos .md a PDF

```bash
python md_to_pdf.py docs/01_fase_analisis.md
python md_to_pdf.py docs/02_fase_diseno.md
# El PDF se genera en la misma carpeta del .md
```

---

## Dataset

**NIH ChestX-ray14** — 112.120 radiografías frontales, 14 patologías, 30.805 pacientes únicos.

- Fuente oficial: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- En Kaggle: [nih-chest-xrays](https://www.kaggle.com/datasets/nih-chest-xrays/data)

> Las imágenes (~42 GB) **no están en este repositorio**. Solo se incluyen los archivos CSV de metadata. Para entrenar, monta el dataset directamente desde Kaggle.
