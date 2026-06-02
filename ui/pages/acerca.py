"""Página: Acerca del proyecto."""

import streamlit as st

from ui.components import section_title


def render():
    section_title("Acerca del proyecto")
    st.markdown("""
    **Proyecto de Aula — Inteligencia Artificial — III Corte**
    Universidad Popular del Cesar · Facultad de Ingeniería de Sistemas · 2026-I

    | | |
    |---|---|
    | **Integrantes** | Mateo Lopez Patiño · Anaclaudia Vega Martinez|
    | **Docente** | Tonny Enrique Jimenez Marquez|
    | **Modelo** | DenseNet-121 con Transfer Learning (ImageNet → NIH ChestX-ray14) |
    | **Dataset** | NIH ChestX-ray14 — 112,120 imágenes · 14 patologías · 30,805 pacientes |
    | **Entrenamiento** | 10 épocas · GPU Tesla T4 (Kaggle) · Adam lr=1e-4 · BCE multi-etiqueta |
    | **Mejor val AUC** | 0.8178 (época 7) · Test AUC medio: 0.7938 |

    ---

    ### Arquitectura del sistema

    ```
    data/          → Metadatos y splits (train/val/test)
    src/           → Pipeline ML (dataset, transforms, modelo, inferencia)
    notebooks/     → EDA, entrenamiento Kaggle, evaluación
    outputs/       → Modelo entrenado, métricas, figuras
    ui/            → Interfaz Streamlit modular
    app.py         → Punto de entrada (router)
    ```

    ### Flujo de procesamiento
    1. **Carga** — usuario sube PNG o JPG (o elige ejemplo NIH).
    2. **Preprocesamiento** — escala de grises → RGB → 224×224 → normalización ImageNet.
    3. **Inferencia** — DenseNet-121 produce 14 probabilidades independientes (sigmoid).
    4. **Grad-CAM** — calculado sobre `denseblock4`, el último bloque convolucional.
    5. **Reporte** — PDF o JSON con imagen, mapa de calor y tabla de resultados.

    ### Umbral de detección
    Configurable en la barra lateral (default **30 %**). Valores más bajos aumentan
    sensibilidad; valores más altos reducen falsos positivos.

    ### Advertencia clínica
    > Este sistema es una herramienta de **apoyo académico**.
    > No reemplaza el criterio de un médico radiólogo especializado.

    ### Referencias
    - Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection.* arXiv:1711.05225
    - Wang et al. (2017). *ChestX-ray8: Hospital-Scale Chest X-Ray Database.* IEEE CVPR.
    - Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
    - Huang et al. (2017). *Densely Connected Convolutional Networks.* IEEE CVPR.
    """)
