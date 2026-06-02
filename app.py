"""
app.py — Sistema de Clasificación de Radiografías de Tórax
Universidad Popular del Cesar — Inteligencia Artificial 2026-I
"""

import warnings

import streamlit as st

from src.config import DEFAULT_UMBRAL
from src.inference.predictor import load_model, load_test_results
from ui.pages import acerca, analisis, eda, evaluacion
from ui.theme import CSS, disclaimer_banner, page_header, sidebar_brand, PALETTE

warnings.filterwarnings("ignore")


def main():
    st.set_page_config(
        page_title="PulmoScan · Radiografías de Tórax",
        page_icon=None,   # sin emoji
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(disclaimer_banner(), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(sidebar_brand(), unsafe_allow_html=True)

        page = st.radio(
            "Navegación",
            [
                "Análisis de imagen",
                "Exploración del dataset",
                "Evaluación del modelo",
                "Acerca del proyecto",
            ],
            label_visibility="collapsed",
        )

        st.markdown(
            f"<hr style='border-color:{PALETTE['border']};margin:1rem 0;'>",
            unsafe_allow_html=True,
        )
        umbral = st.slider(
            "Filtro visual de resultados",
            min_value=0.10, max_value=0.60, value=DEFAULT_UMBRAL, step=0.05,
            help="Solo resalta patologías con probabilidad ≥ este valor. "
                 "No modifica el modelo ni sus probabilidades.",
        )
        st.caption(
            f"Filtro de visualización ≥ {umbral:.0%}. "
            "Las probabilidades del modelo no cambian."
        )

        sample_path = None
        if "Análisis" in page:
            sample_path = analisis.render_sample_picker()

        st.markdown(f"""
        <hr style='border-color:{PALETTE['border']};margin-top:1.5rem;'>
        <div style='font-size:0.68rem;color:{PALETTE['text_light']};text-align:center;
                    margin-top:0.6rem;line-height:1.65;'>
          DenseNet-121 · NIH ChestX-ray14<br>
          UPC · Inteligencia Artificial 2026-I
        </div>
        """, unsafe_allow_html=True)

    model, ckpt = load_model()
    test_results = load_test_results()

    chips = []
    if ckpt:
        chips = [
            ("Modelo", "DenseNet-121"),
            ("Época", f"{ckpt.get('epoch', '?')} / 10"),
            ("Val AUC", f"{ckpt.get('val_auc', 0):.4f}"),
            ("Clases", "14 patologías"),
        ]

    st.markdown(
        page_header(
            "Clasificación de radiografías de tórax",
            "Detección multi-etiqueta de patologías pulmonares con deep learning "
            "y mapas de atención Grad-CAM.",
            chips=chips if chips else None,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<hr style='border-color:{PALETTE['border_light']};margin:0.5rem 0 1.25rem;'>",
        unsafe_allow_html=True,
    )

    if model is None and "Análisis" in page:
        st.error(
            "No se encontró `outputs/models/best_model.pt`. "
            "Descárgalo de Kaggle y colócalo en esa ruta."
        )
        return

    if "Análisis" in page:
        analisis.render(model, ckpt, test_results, umbral, sample_path)
    elif "Exploración" in page:
        eda.render()
    elif "Evaluación" in page:
        evaluacion.render(test_results)
    else:
        acerca.render()


if __name__ == "__main__":
    main()