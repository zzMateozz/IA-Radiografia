"""Página: Evaluación de resultados del modelo."""

import pandas as pd
import streamlit as st

from src.config import CHEXNET_AUC, EVAL_FIGURES, PATHOLOGIES
from ui.components import section_title, show_figure


def render(test_results):
    section_title(
        "📊", "Evaluación de Resultados",
        "Métricas del modelo sobre el set de test de NIH ChestX-ray14 (25,596 imágenes).",
    )

    if test_results is None:
        st.warning(
            "No se encontró `outputs/models/test_results.json`. "
            "Ejecuta primero el notebook `03_evaluacion_resultados.ipynb`."
        )
        return

    auc_mean = test_results["test_auc_mean"]
    best_ep = test_results["best_epoch"]
    val_auc = test_results["best_val_auc"]
    chex_mean = sum(CHEXNET_AUC.values()) / len(CHEXNET_AUC)

    cols = st.columns(4, gap="small")
    with cols[0]:
        st.metric("Test AUC promedio", f"{auc_mean:.4f}")
    with cols[1]:
        st.metric("CheXNet AUC (ref.)", f"{chex_mean:.4f}")
    with cols[2]:
        st.metric("Mejor época", f"{best_ep} / 10")
    with cols[3]:
        st.metric("Val AUC (mejor época)", f"{val_auc:.4f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Curvas de aprendizaje",
        "🏆 AUC por patología",
        "📈 Curvas ROC",
        "🔍 Distribución de predicciones",
    ])

    with tab1:
        show_figure(EVAL_FIGURES[0])

    with tab2:
        rows = []
        for p in PATHOLOGIES:
            our = test_results["test_auc_per_class"][p]
            ref = CHEXNET_AUC[p]
            rows.append({
                "Patología": p,
                "Nuestro AUC": round(our, 4),
                "CheXNet AUC": round(ref, 4),
                "Δ Diferencia": round(our - ref, 4),
                "F1-Score": round(test_results["test_f1_per_class"][p], 4),
                "Avg Precision": round(test_results["test_ap_per_class"][p], 4),
            })
        df = pd.DataFrame(rows)

        def hl(v):
            return (
                "background-color:#052e16;color:#4ade80;font-weight:600"
                if v >= 0 else
                "background-color:#450a0a;color:#f87171;font-weight:600"
            )

        styled = (
            df.style
            .map(hl, subset=["Δ Diferencia"])
            .format({
                "Nuestro AUC": "{:.4f}",
                "CheXNet AUC": "{:.4f}",
                "Δ Diferencia": "{:+.4f}",
                "F1-Score": "{:.4f}",
                "Avg Precision": "{:.4f}",
            })
        )
        st.dataframe(styled, use_container_width=True, height=530)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        show_figure(EVAL_FIGURES[1])

    with tab3:
        show_figure(EVAL_FIGURES[2])

    with tab4:
        show_figure(EVAL_FIGURES[3])
