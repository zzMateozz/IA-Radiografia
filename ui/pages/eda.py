"""Página: EDA del dataset NIH — versión limpia y profesional."""

import streamlit as st

from src.config import EDA_FIGURES
from ui.components import card, section_title, show_figure


def render():
    section_title(
        "Exploración del dataset NIH",
        "NIH ChestX-ray14 — 112 120 radiografías · 14 patologías · 30 805 pacientes",
    )

    cols = st.columns(4, gap="small")
    stats = [
        ("Imágenes totales", "112 120", "radiografías frontales"),
        ("Pacientes únicos", "30 805", "sin solapamiento train/test"),
        ("Patologías", "14", "clasificación multi-etiqueta"),
        ("Sin hallazgo", "53.8 %", "60 361 imágenes normales"),
    ]
    for col, (lbl, val, sub) in zip(cols, stats):
        with col:
            st.markdown(card(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Distribución de patologías",
        "Demografía y pacientes",
        "Co‑ocurrencia y edad",
    ])

    with tab1:
        show_figure(EDA_FIGURES[0])
        st.markdown("---")
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            show_figure(EDA_FIGURES[1])
        with col_b:
            show_figure(EDA_FIGURES[2])

    with tab2:
        show_figure(EDA_FIGURES[4])
        st.markdown("---")
        show_figure(EDA_FIGURES[6])

    with tab3:
        show_figure(EDA_FIGURES[3])
        st.markdown("---")
        show_figure(EDA_FIGURES[5])