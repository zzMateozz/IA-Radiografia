"""Página: Análisis de imagen + Grad-CAM + PDF."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import PATHOLOGIES, list_sample_images
from src.inference.gradcam import make_gradcam_overlay, ndarray_to_pil
from src.inference.predictor import predict
from src.reports.pdf_builder import build_pdf
from ui.components import badge, hero_summary, model_cards, prob_bar, section_title


def _empty_state():
    st.markdown("""
    <div style='border:2px dashed #1a3a6b;border-radius:16px;padding:4rem;
                text-align:center;color:#4d6a99;background:#0d1f3c;margin-top:1rem;'>
      <div style='font-size:3.5rem;margin-bottom:0.5rem;'>🫁</div>
      <div style='font-size:1.05rem;font-weight:600;color:#60a5fa;'>
        Arrastra o selecciona una radiografía
      </div>
      <div style='font-size:0.82rem;margin-top:0.4rem;'>PNG · JPG · JPEG</div>
      <div style='font-size:0.78rem;margin-top:0.8rem;color:#64748b;'>
        O usa un ejemplo del dataset NIH en la barra lateral
      </div>
    </div>""", unsafe_allow_html=True)


def _render_results(probs, umbral: float, test_results):
    section_title("📋", "Resultados — 14 patologías")
    positivos = sorted(
        [(p, probs[i]) for i, p in enumerate(PATHOLOGIES) if probs[i] >= umbral],
        key=lambda x: x[1], reverse=True,
    )
    negativos = sorted(
        [(p, probs[i]) for i, p in enumerate(PATHOLOGIES) if probs[i] < umbral],
        key=lambda x: x[1], reverse=True,
    )

    col_pos, col_neg = st.columns([1.1, 1], gap="large")
    with col_pos:
        if positivos:
            n_a = sum(1 for _, v in positivos if v >= 0.5)
            n_m = sum(1 for _, v in positivos if umbral <= v < 0.5)
            st.markdown(
                f"<div style='margin-bottom:1rem;'>"
                f"{badge(f'{n_a} alta(s)', '#ef4444')} &nbsp;"
                f"{badge(f'{n_m} media(s)', '#f97316')}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("".join(
                prob_bar(p, v, umbral, test_results=test_results) for p, v in positivos
            ), unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#052e16;border:1px solid #166534;
                        border-radius:12px;padding:1.5rem;text-align:center;'>
              <div style='font-size:1.8rem;'>✅</div>
              <div style='color:#4ade80;font-weight:600;margin-top:0.4rem;'>
                Sin hallazgos detectados
              </div>
              <div style='color:#166534;font-size:0.82rem;margin-top:0.3rem;'>
                Todas las probabilidades &lt; {umbral:.0%}
              </div>
            </div>""", unsafe_allow_html=True)
    with col_neg:
        st.markdown(
            "<div style='color:#4d6a99;font-size:0.82rem;margin-bottom:0.8rem;'>"
            "Patologías no detectadas</div>",
            unsafe_allow_html=True,
        )
        st.markdown("".join(
            prob_bar(p, v, umbral, show_info=False) for p, v in negativos
        ), unsafe_allow_html=True)


def render(model, ckpt, test_results, umbral: float, sample_path: Path | None = None):
    section_title(
        "🔬", "Análisis de imagen",
        "Sube una radiografía de tórax (PNG · JPG) para obtener predicción "
        "de 14 patologías y mapa de calor Grad-CAM.",
    )

    uploaded = st.file_uploader(
        "Radiografía", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
    )

    img_source = None
    filename = "radiografia.png"

    if uploaded is not None:
        img_source = uploaded
        filename = uploaded.name
    elif sample_path is not None and sample_path.exists():
        img_source = sample_path
        filename = sample_path.name
        st.info(f"📎 Ejemplo cargado: `{filename}`")

    if img_source is None:
        _empty_state()
        return

    img_pil = Image.open(img_source).convert("RGB")

    with st.status("Analizando radiografía…", expanded=True) as status:
        st.write("Preprocesando imagen (224×224, normalización ImageNet)…")
        probs, img_tensor = predict(model, img_pil)
        st.write("Calculando 14 probabilidades (sigmoid multi-etiqueta)…")

        top_idx = int(probs.argmax())
        with st.expander("🎯 Seleccionar patología para Grad-CAM", expanded=False):
            selected_name = st.radio(
                "p", PATHOLOGIES, index=top_idx,
                horizontal=True, label_visibility="collapsed",
            )
        target_idx = PATHOLOGIES.index(selected_name)

        st.write(f"Generando Grad-CAM para **{selected_name}**…")
        img_rgb, _cam_arr, overlay_arr = make_gradcam_overlay(
            model, img_tensor, target_idx)
        pil_orig = ndarray_to_pil(img_rgb)
        pil_overlay = ndarray_to_pil(overlay_arr)
        status.update(label="Análisis completado", state="complete")

    hero_summary(probs, umbral)

    st.markdown("---")
    st.markdown(
        f"<h4 style='margin-bottom:0.6rem;'>🔥 Grad-CAM — "
        f"<span style='color:#60a5fa;'>{selected_name}</span> "
        f"<span style='color:#f97316;font-family:\"JetBrains Mono\",monospace;"
        f"font-size:1rem;'>({probs[target_idx]:.1%})</span></h4>",
        unsafe_allow_html=True,
    )
    col_orig, col_cam = st.columns(2, gap="medium")
    with col_orig:
        st.markdown(
            "<p style='text-align:center;color:#4d6a99;font-size:0.85rem;"
            "margin-bottom:6px;'>📷 Radiografía original</p>",
            unsafe_allow_html=True,
        )
        st.image(pil_orig, use_container_width=True)
    with col_cam:
        st.markdown(
            f"<p style='text-align:center;color:#4d6a99;font-size:0.85rem;"
            f"margin-bottom:6px;'>🌡 Grad-CAM · {selected_name}</p>",
            unsafe_allow_html=True,
        )
        st.image(pil_overlay, use_container_width=True)

    st.markdown("""
    <div style='background:#0d1f3c;border:1px solid #1a3a6b;border-radius:10px;
                padding:0.8rem 1.2rem;font-size:0.8rem;color:#4d6a99;margin-top:0.5rem;'>
      <b style='color:#e2e8f0;'>¿Cómo leer el mapa?</b> &nbsp;
      Zonas en <span style='color:#ef4444;'>■ rojo</span> = región más relevante
      para la predicción. Zonas en <span style='color:#3b82f6;'>■ azul</span> = baja
      relevancia. Generado con Grad-CAM sobre denseblock4 de DenseNet-121.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    _render_results(probs, umbral, test_results)

    st.markdown("---")
    section_title("🧠", "Información del modelo")
    model_cards(ckpt, test_results)

    st.markdown("---")
    section_title("📄", "Exportar resultados",
                  "Descarga el reporte en PDF o los datos en JSON.")

    col_pdf, col_json = st.columns(2)
    with col_pdf:
        with st.spinner("Preparando PDF…"):
            pdf_bytes = build_pdf(
                img_pil, pil_overlay, probs, filename, ckpt or {}, test_results, umbral)
        st.download_button(
            "⬇  Descargar reporte PDF",
            data=pdf_bytes,
            file_name=f"reporte_rx_{Path(filename).stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col_json:
        export = {
            "archivo": filename,
            "umbral": umbral,
            "predicciones": {
                PATHOLOGIES[i]: round(float(probs[i]), 4) for i in range(len(PATHOLOGIES))
            },
            "gradcam_patologia": selected_name,
        }
        st.download_button(
            "⬇  Descargar JSON",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name=f"resultados_{Path(filename).stem}.json",
            mime="application/json",
            use_container_width=True,
        )


def render_sample_picker() -> Path | None:
    """Selector de ejemplos en sidebar. Retorna path si el usuario elige uno."""
    samples = list_sample_images()
    if not samples:
        return None
    st.markdown(
        "<div style='font-size:0.75rem;color:#4d6a99;margin:0.5rem 0 0.3rem;'>"
        "Ejemplos NIH</div>",
        unsafe_allow_html=True,
    )
    choice = st.selectbox(
        "sample",
        options=["— Subir mi propia imagen —"] + [p.name for p in samples],
        label_visibility="collapsed",
    )
    if choice == "— Subir mi propia imagen —":
        return None
    return next(p for p in samples if p.name == choice)
