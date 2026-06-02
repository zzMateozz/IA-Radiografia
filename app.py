"""
app.py — Sistema de Clasificación de Radiografías de Tórax
Universidad Popular del Cesar — Inteligencia Artificial 2026-I

Ejecutar:
    streamlit run app.py
"""

import warnings

import streamlit as st

from src.config import DEFAULT_UMBRAL
from src.inference.predictor import load_model, load_test_results
from ui.pages import acerca, analisis, eda, evaluacion
from ui.theme import CSS, disclaimer_banner

warnings.filterwarnings("ignore")


def main():
    st.set_page_config(
        page_title="RX-IA · Tórax",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(disclaimer_banner(), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1.5rem 0 1.2rem;'>
          <div style='font-size:3.2rem;'>🫁</div>
          <div style='font-family:"JetBrains Mono",monospace;font-size:1.25rem;
                      font-weight:700;color:#60a5fa;letter-spacing:3px;margin-top:6px;'>
            RX-IA
          </div>
          <div style='font-size:0.7rem;color:#4d6a99;margin-top:4px;
                      letter-spacing:1px;text-transform:uppercase;'>
            Clasificación de Tórax
          </div>
        </div>
        <hr style='border-color:#1a2640;margin-bottom:1.2rem;'>
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            [
                "🔬  Análisis de imagen",
                "📈  EDA — Dataset NIH",
                "📊  Evaluación de resultados",
                "ℹ️  Acerca del proyecto",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#1a2640;margin:1rem 0;'>", unsafe_allow_html=True)
        umbral = st.slider(
            "Umbral de detección",
            min_value=0.10, max_value=0.60, value=DEFAULT_UMBRAL, step=0.05,
            help="Probabilidad mínima para marcar un hallazgo como presente.",
        )
        st.caption(f"Patologías con prob ≥ {umbral:.0%} se marcan como detectadas.")

        sample_path = None
        if "Análisis" in page:
            sample_path = analisis.render_sample_picker()

        st.markdown("""
        <hr style='border-color:#1a2640;margin-top:1.5rem;'>
        <div style='font-size:0.68rem;color:#1a3a6b;text-align:center;
                    margin-top:0.8rem;line-height:1.7;'>
          DenseNet-121 · NIH ChestX-ray14<br>
          Universidad Popular del Cesar<br>
          Inteligencia Artificial · 2026-I
        </div>
        """, unsafe_allow_html=True)

    model, ckpt = load_model()
    test_results = load_test_results()

    st.markdown("""
    <div style='margin-bottom:0.2rem;'>
      <span style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;
                   color:#1d4ed8;letter-spacing:4px;text-transform:uppercase;'>
        Universidad Popular del Cesar · IA 2026-I
      </span>
    </div>
    <h1 style='font-size:1.9rem;font-weight:700;margin:0 0 0.1rem;color:#f1f5f9;
               font-family:"Space Grotesk",sans-serif;line-height:1.2;'>
      Sistema de Clasificación de<br>Radiografías de Tórax
    </h1>
    """, unsafe_allow_html=True)

    if ckpt:
        st.markdown(
            f"<div style='display:flex;gap:1.5rem;margin:0.5rem 0 0.2rem;"
            f"font-size:0.76rem;font-family:\"JetBrains Mono\",monospace;color:#1d4ed8;'>"
            f"<span>DenseNet-121</span>"
            f"<span>Época <span style='color:#60a5fa;'>{ckpt.get('epoch', '?')} / 10</span></span>"
            f"<span>Val AUC <span style='color:#60a5fa;'>{ckpt.get('val_auc', 0):.4f}</span></span>"
            f"<span>14 patologías · NIH ChestX-ray14</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1a2640;margin:0.8rem 0 1rem;'>", unsafe_allow_html=True)

    if model is None and "Análisis" in page:
        st.error(
            "No se encontró `outputs/models/best_model.pt`. "
            "Descárgalo de Kaggle y colócalo en esa ruta."
        )
        return

    if "Análisis" in page:
        analisis.render(model, ckpt, test_results, umbral, sample_path)
    elif "EDA" in page:
        eda.render()
    elif "Evaluación" in page:
        evaluacion.render(test_results)
    else:
        acerca.render()


if __name__ == "__main__":
    main()
