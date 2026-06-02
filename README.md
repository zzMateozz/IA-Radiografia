# PulmoScan — Clasificación de Radiografías de Tórax (NIH ChestX-ray14)

**Proyecto de Aula — Inteligencia Artificial — III Corte**  
Universidad Popular del Cesar — Facultad de Ingeniería de Sistemas — 2026-I

**Integrantes:**
- Mateo Lopez Patiño
- Anaclaudia Vega Martinez

**Docente:**
- Tonny Enrique Jimenez Marquez

---

## ¿Qué hace este proyecto?

Clasifica automáticamente **14 patologías torácicas** a partir de radiografías de tórax usando una red neuronal **DenseNet-121** (estilo CheXNet) entrenada sobre el dataset público **NIH ChestX-ray14** (112.120 imágenes).

Incluye una aplicación web **PulmoScan** (Streamlit) para:

- Subir radiografías y obtener **probabilidades** por patología
- Visualizar **Grad-CAM** (mapa de atención del modelo)
- **Comparar** predicciones con la etiqueta oficial NIH (si la imagen es del dataset)
- Ver métricas de evaluación, gráficas del EDA y exportar **PDF / JSON**

> **Aviso:** Herramienta de uso exclusivamente académico. No reemplaza el criterio de un médico radiólogo.

---

## Inicio rápido (demo Streamlit)

```powershell
# 1. Clonar e instalar (ver sección completa abajo si es la primera vez)
git clone https://github.com/zzMateozz/IA-Radiografia.git
cd IA-Radiografia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Verificar que exista el modelo entrenado
#    Debe estar en: outputs/models/best_model.pt

# 3. Ejecutar la app
streamlit run app.py
```

La app abre en el navegador (por defecto `http://localhost:8501`).

**Requisito obligatorio:** `outputs/models/best_model.pt` (pesos del entrenamiento en Kaggle). Sin ese archivo la página de análisis no carga el modelo.

---

## Requisitos previos

| Herramienta | Versión mínima | Cómo verificar |
|---|---|---|
| Python | 3.10 o superior | `python --version` |
| Git | cualquiera | `git --version` |
| VSCode | cualquiera | (con extensión **Jupyter** instalada) |

> **Nota GPU:** El entrenamiento requiere GPU NVIDIA (CUDA). Si solo tienes CPU o GPU AMD, puedes correr el EDA, la evaluación y la demo localmente; el entrenamiento se hace en **Kaggle** (gratis, Tesla T4).

---

## Paso a paso: configuración desde cero

### 1. Clonar el repositorio

```bash
git clone https://github.com/zzMateozz/IA-Radiografia.git
cd IA-Radiografia
```

### 2. Crear el entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

**Mac / Linux:**
```bash
python3 -m venv .venv
```

### 3. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea scripts:
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

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Instala PyTorch, Streamlit, pandas, scikit-learn, ReportLab y el resto. Tarda ~3-5 minutos la primera vez.

### 5. Registrar el kernel de Jupyter para VSCode

```bash
python -m ipykernel install --user --name=proyecto_fase3 --display-name="Python (proyecto_fase3)"
```

### 6. Abrir en VSCode

```bash
code .
```

### 7. Seleccionar el kernel en el notebook

1. Abre cualquier notebook en `notebooks/`
2. Esquina superior derecha → selector de kernel
3. Selecciona **Python (proyecto_fase3)** o el entorno `.venv`

---

## Ejecutar PulmoScan (app Streamlit)

Con el entorno virtual activado y desde la raíz del proyecto:

```bash
streamlit run app.py
```

### Páginas de la aplicación

| Página | Qué muestra |
|---|---|
| **Análisis de imagen** | Subir radiografía, predicciones, Grad-CAM, comparación NIH, exportar PDF/JSON |
| **Exploración del dataset** | 7 figuras del análisis exploratorio (EDA) |
| **Evaluación del modelo** | AUC por patología, curvas ROC, comparación con CheXNet |
| **Acerca del proyecto** | Equipo, referencias y disclaimer |

### Cómo probar con imágenes NIH

1. En el sidebar, sección **Ejemplos NIH**, elige una radiografía de muestra, **o**
2. Sube un PNG conservando el **nombre original** del dataset (ej. `00000004_000.png`)

Si el nombre coincide con una fila de `data/processed/metadata_clean.csv`, aparece la sección **Comparación con etiqueta NIH**.

### Filtro visual del sidebar

El slider **“Filtro visual de resultados”** solo organiza qué patologías se resaltan abajo (probabilidad ≥ valor elegido). **No modifica** las predicciones del modelo ni la comparación NIH.

---

## Funcionalidades implementadas (demo)

### Predicción multi-etiqueta

El modelo devuelve **14 probabilidades** independientes (0 a 1), una por patología. No elige una sola enfermedad.

### Grad-CAM

Mapa de calor sobre la radiografía que indica **en qué regiones** el modelo se basó para una patología seleccionada. Implementado en `src/inference/gradcam.py` sobre el bloque `denseblock4` de DenseNet-121.

### Comparación con etiqueta NIH

Para imágenes del dataset, la app consulta `data/processed/metadata_clean.csv` y muestra:

- Etiqueta oficial NIH (`labels_raw`)
- Probabilidad del modelo por cada patología
- Notas interpretativas (coherencia alta/moderada/baja)
- Alertas si hay probabilidad ≥ 50% en patologías no etiquetadas por NIH

Implementado en `src/inference/ground_truth.py`.

### Exportar resultados

Desde **Análisis de imagen**:

- **PDF:** reporte con radiografía, Grad-CAM, probabilidades y comparación NIH (si aplica)
- **JSON:** predicciones crudas + metadatos de comparación

---

## Estructura del proyecto

```
IA-Radiografia/
├── app.py                          # App Streamlit (PulmoScan)
├── requirements.txt
├── md_to_pdf.py                    # Convierte informes .md a PDF
│
├── data/
│   ├── raw/                        # CSVs originales del NIH
│   ├── processed/                  # metadata_clean.csv, metadata_with_splits.csv
│   └── sample_images/              # Radiografías de ejemplo para la demo
│
├── docs/
│   ├── GUIA_ESTUDIO_COMPLETA.md    # Guía de estudio y sustentación
│   ├── 01_fase_analisis.md
│   ├── 02_fase_diseno.md
│   └── 03_fase_prueba.md
│
├── notebooks/
│   ├── 01_eda_metadata.ipynb       # EDA (local, CPU)
│   ├── 02_kaggle_training.ipynb    # Entrenamiento (Kaggle, GPU)
│   └── 03_evaluacion_resultados.ipynb  # Evaluación y gráficas
│
├── outputs/
│   ├── figures/                    # Gráficas EDA + evaluación
│   └── models/
│       ├── best_model.pt           # Pesos del modelo (requerido para la app)
│       ├── history.json
│       └── test_results.json
│
├── src/
│   ├── config.py                   # Constantes compartidas (ML + UI)
│   ├── data/                       # Dataset, splits, transforms
│   ├── models/                     # DenseNet-121 (CheXNet)
│   ├── inference/                  # predictor, Grad-CAM, ground_truth NIH
│   └── reports/                    # Generación de PDF
│
└── ui/                             # Interfaz Streamlit (páginas, tema, componentes)
    ├── pages/
    │   ├── analisis.py
    │   ├── eda.py
    │   ├── evaluacion.py
    │   └── acerca.py
    ├── components.py
    └── theme.py
```

---

## Flujo de trabajo del proyecto

```
[1] EDA local           → notebooks/01_eda_metadata.ipynb
[2] Build splits        → python -m src.data.build_splits (ya generado)
[3] Entrenamiento       → notebooks/02_kaggle_training.ipynb en Kaggle
[4] Descargar modelo    → copiar best_model.pt a outputs/models/
[5] Evaluación local    → notebooks/03_evaluacion_resultados.ipynb
[6] Demo PulmoScan      → streamlit run app.py
```

---

## Estrategia de cómputo

| Tarea | Entorno | Razón |
|---|---|---|
| EDA y preprocesamiento | PC local (CPU) | Solo metadata, no necesita GPU |
| Entrenamiento del modelo | Kaggle (Tesla T4 gratis) | GPU NVIDIA necesaria (CUDA) |
| Evaluación e inferencia | PC local (CPU) | Modelo ya entrenado |
| Demo Streamlit | PC local (CPU) | Inferencia sobre imagen individual |

---

## Resultados clave del modelo

| Métrica | Valor |
|---|---|
| Arquitectura | DenseNet-121 + Transfer Learning (ImageNet) |
| Test AUC promedio | **0.7938** |
| CheXNet (referencia) | 0.8399 |
| Mejor época | 7 / 10 |
| Val AUC (mejor época) | 0.8178 |

Detalle por patología en `docs/03_fase_prueba.md` y en la página **Evaluación del modelo** de la app.

---

## Convertir documentos .md a PDF

```bash
python md_to_pdf.py docs/01_fase_analisis.md
python md_to_pdf.py docs/02_fase_diseno.md
python md_to_pdf.py docs/03_fase_prueba.md
# El PDF se genera en la misma carpeta del .md
```

---

## Dataset

**NIH ChestX-ray14** — 112.120 radiografías frontales, 14 patologías, 30.805 pacientes únicos.

- Fuente oficial: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- En Kaggle: [nih-chest-xrays](https://www.kaggle.com/datasets/nih-chest-xrays/data)

> Las imágenes (~42 GB) **no están en este repositorio**. Se incluyen los CSV de metadata y radiografías de ejemplo en `data/sample_images/`. Para entrenar, monta el dataset completo desde Kaggle.

---

## Tecnologías principales

| Tecnología | Uso en el proyecto |
|---|---|
| **Python** | Lenguaje base |
| **PyTorch / Torchvision** | Modelo, entrenamiento e inferencia |
| **DenseNet-121 (CheXNet)** | Arquitectura CNN multi-etiqueta |
| **NIH ChestX-ray14** | Dataset de entrenamiento y evaluación |
| **Grad-CAM** | Interpretabilidad en la demo |
| **Kaggle (GPU)** | Entrenamiento con Tesla T4 |
| **Streamlit** | Interfaz PulmoScan |

---

## Solución de problemas

| Problema | Solución |
|---|---|
| `No se encontró best_model.pt` | Copia el archivo desde Kaggle a `outputs/models/best_model.pt` |
| No aparece comparación NIH | El nombre del archivo debe coincidir con `image_id` en `metadata_clean.csv` |
| PowerShell no activa `.venv` | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Streamlit no abre | Verifica que el entorno esté activo y ejecuta `pip install streamlit` |

---

## Documentación adicional

- **Guía completa de estudio y sustentación:** `docs/GUIA_ESTUDIO_COMPLETA.md`
- **Informes por fase:** `docs/01_fase_analisis.md`, `02_fase_diseno.md`, `03_fase_prueba.md`
