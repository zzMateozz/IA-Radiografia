"""Página: Análisis de imagen + Grad-CAM + PDF."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import PATHOLOGIES, list_sample_images
from src.inference.gradcam import make_gradcam_overlay, ndarray_to_pil
from src.inference.ground_truth import get_ground_truth, summarize_nih_comparison
from src.inference.predictor import predict
from src.reports.pdf_builder import build_pdf
from ui.components import badge, hero_summary, model_cards, prob_bar, section_title
from ui.theme import PALETTE as C


def _empty_state():
    st.markdown(f"""
    <div style='border:2px dashed {C['primary_light']};border-radius:20px;padding:3.5rem 2rem;
                text-align:center;background:{C['surface']};margin-top:0.5rem;
                box-shadow:{C['shadow_sm']};'>
      <div style='font-size:1.1rem;font-weight:600;color:{C['text']};'>
        Sube una radiografía para comenzar
      </div>
      <div style='font-size:0.88rem;color:{C['text_muted']};margin-top:0.5rem;'>
        Formatos PNG, JPG o JPEG · máx. recomendado 10 MB
      </div>
      <div style='font-size:0.8rem;color:{C['text_light']};margin-top:0.75rem;'>
        También puedes elegir un ejemplo del dataset NIH en la barra lateral
      </div>
    </div>""", unsafe_allow_html=True)


def _render_ground_truth_comparison(probs, filename: str):
    gt = get_ground_truth(filename)
    if gt is None:
        return None

    cmp = summarize_nih_comparison(probs, gt)
    nih_text = gt["labels_raw"] if not gt["no_finding"] else "No Finding"

    section_title(
        "Comparación con etiqueta NIH",
        "Etiqueta oficial del dataset vs probabilidades del modelo (sin umbral binario).",
    )

    if cmp["n_nih_labeled"]:
        summary = (
            f"NIH etiqueta {cmp['n_nih_labeled']} patología(s). "
            f"Con probabilidad ≥ 50%: {cmp['n_nih_labeled_high']}/{cmp['n_nih_labeled']}."
        )
        summary_kind = "success" if cmp["n_nih_labeled_high"] == cmp["n_nih_labeled"] else "neutral"
    else:
        summary = "NIH: No Finding — sin patologías etiquetadas."
        summary_kind = "neutral"

    st.markdown(
        f"<div style='margin-bottom:0.75rem;'>{badge(summary, summary_kind)}</div>",
        unsafe_allow_html=True,
    )

    col_nih, col_model = st.columns(2, gap="medium")
    with col_nih:
        st.markdown(
            f"<div style='background:{C['surface_soft']};border:1px solid {C['border']};"
            f"border-radius:12px;padding:1rem 1.1rem;'>"
            f"<div style='font-size:0.78rem;font-weight:600;color:{C['text_muted']};"
            f"text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.4rem;'>"
            f"Etiqueta NIH</div>"
            f"<div style='font-size:0.95rem;font-weight:600;color:{C['text']};'>"
            f"{nih_text}</div>"
            f"<div style='font-size:0.8rem;color:{C['text_muted']};margin-top:0.35rem;'>"
            f"Archivo: {gt['image_id']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_model:
        top_lines = "<br>".join(
            f"{r['pathology']}: <b>{r['prob']:.1%}</b>"
            for r in cmp["top_probs"]
        )
        st.markdown(
            f"<div style='background:{C['surface_soft']};border:1px solid {C['border']};"
            f"border-radius:12px;padding:1rem 1.1rem;'>"
            f"<div style='font-size:0.78rem;font-weight:600;color:{C['text_muted']};"
            f"text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.4rem;'>"
            f"Mayores probabilidades del modelo</div>"
            f"<div style='font-size:0.88rem;color:{C['text']};line-height:1.7;'>"
            f"{top_lines}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    nih_first = sorted(cmp["rows"], key=lambda r: (not r["nih"], -r["prob"]))
    with st.expander("Ver detalle por patología", expanded=True):
        for row in nih_first:
            nih_lbl = "Sí" if row["nih"] else "No"
            nih_style = (
                f"background:{C['primary']}14;border-left:3px solid {C['primary']};"
                if row["nih"] else ""
            )
            note_badge = badge(row["note"], row["note_kind"])
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"flex-wrap:wrap;gap:0.5rem;padding:0.5rem 0.65rem;margin-bottom:0.25rem;"
                f"border-bottom:1px solid {C['border_light']};font-size:0.85rem;"
                f"border-radius:8px;{nih_style}'>"
                f"<span style='font-weight:600;color:{C['text']};min-width:140px;'>"
                f"{row['pathology']}</span>"
                f"<span style='color:{C['text_muted']};'>NIH: <b>{nih_lbl}</b></span>"
                f"<span style='font-weight:700;color:{C['text']};'>{row['prob']:.1%}</span>"
                f"{note_badge}"
                f"</div>",
                unsafe_allow_html=True,
            )

    if cmp["high_without_label"]:
        names = ", ".join(r["pathology"] for r in cmp["high_without_label"])
        st.info(
            f"Probabilidad ≥ 50% sin etiqueta NIH: **{names}**. "
            "Puede ser un falso positivo o una etiqueta incompleta del dataset."
        )

    st.markdown(f"""
    <div style='background:{C['surface_soft']};border:1px solid {C['border_light']};
                border-radius:12px;padding:0.75rem 1rem;font-size:0.8rem;
                color:{C['text_muted']};margin-top:0.5rem;line-height:1.5;'>
      El modelo entrega <b>probabilidades</b>, no diagnóstico sí/no.
      Las etiquetas NIH provienen de <code>metadata_clean.csv</code>.
      El filtro del sidebar solo afecta la sección de resultados, no esta comparación.
    </div>""", unsafe_allow_html=True)

    return cmp


def _render_results(probs, umbral: float, test_results):
    section_title(
        "Resultados por patología",
        f"Patologías con probabilidad ≥ {umbral:.0%} (filtro visual del sidebar).",
    )
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
                f"<div style='margin-bottom:1rem;display:flex;gap:8px;flex-wrap:wrap;'>"
                f"{badge(f'{n_a} confianza alta', 'high')} "
                f"{badge(f'{n_m} confianza media', 'medium')}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("".join(
                prob_bar(p, v, umbral, test_results=test_results) for p, v in positivos
            ), unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#D1E7DD;border:1px solid #A3CFBB;border-radius:12px;
                        padding:1.75rem;text-align:center;box-shadow:{C['shadow_sm']};'>
              <div style='color:{C['success']};font-weight:600;font-size:1rem;'>
                Sin hallazgos sobre el umbral
              </div>
              <div style='color:{C['text_muted']};font-size:0.85rem;margin-top:0.35rem;'>
                Todas las probabilidades están por debajo del filtro ({umbral:.0%})
              </div>
            </div>""", unsafe_allow_html=True)
    with col_neg:
        st.markdown(
            f"<div style='color:{C['text_muted']};font-size:0.85rem;margin-bottom:0.9rem;"
            f"font-weight:500;'>Por debajo del filtro visual ({umbral:.0%})</div>",
            unsafe_allow_html=True,
        )
        st.markdown("".join(
            prob_bar(p, v, umbral, show_info=False) for p, v in negativos
        ), unsafe_allow_html=True)


def render(model, ckpt, test_results, umbral: float, sample_path: Path | None = None):
    section_title(
        "Análisis de imagen",
        "Carga una radiografía para obtener predicciones de 14 patologías "
        "y visualizar las regiones que el modelo considera relevantes.",
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
        st.markdown(
            f"<div style='background:{C['surface_soft']};border:1px solid {C['border']};"
            f"border-radius:10px;padding:0.65rem 1rem;font-size:0.85rem;"
            f"color:{C['text_muted']};margin-bottom:0.5rem;'>"
            f"Ejemplo del dataset NIH: <b style='color:{C['primary']};'>{filename}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if img_source is None:
        _empty_state()
        return

    img_pil = Image.open(img_source).convert("RGB")

    with st.status("Procesando radiografía…", expanded=True) as status:
        st.write("Redimensionando y normalizando la imagen…")
        probs, img_tensor = predict(model, img_pil)
        st.write("Calculando probabilidades para las 14 patologías…")

        top_idx = int(probs.argmax())
        with st.expander("Elegir patología para el mapa Grad-CAM", expanded=False):
            selected_name = st.radio(
                "Selecciona la patología a visualizar", PATHOLOGIES, index=top_idx,
                horizontal=True, label_visibility="collapsed",
            )
        target_idx = PATHOLOGIES.index(selected_name)

        st.write(f"Generando mapa de atención para **{selected_name}**…")
        img_rgb, _cam_arr, overlay_arr = make_gradcam_overlay(
            model, img_tensor, target_idx)
        pil_orig = ndarray_to_pil(img_rgb)
        pil_overlay = ndarray_to_pil(overlay_arr)
        status.update(label="Análisis completado", state="complete")

    hero_summary(probs, umbral)
    comparison = _render_ground_truth_comparison(probs, filename)

    section_title(
        f"Mapa Grad-CAM — {selected_name}",
        f"Probabilidad del modelo: {probs[target_idx]:.1%}. "
        "Las zonas cálidas indican regiones con mayor influencia en la predicción.",
    )
    col_orig, col_cam = st.columns(2, gap="medium")
    with col_orig:
        st.markdown(
            f"<p style='text-align:center;color:{C['text_muted']};font-size:0.85rem;"
            f"margin-bottom:8px;font-weight:500;'>Radiografía original</p>",
            unsafe_allow_html=True,
        )
        st.image(pil_orig, use_container_width=True)
    with col_cam:
        st.markdown(
            f"<p style='text-align:center;color:{C['text_muted']};font-size:0.85rem;"
            f"margin-bottom:8px;font-weight:500;'>Mapa de atención · {selected_name}</p>",
            unsafe_allow_html=True,
        )
        st.image(pil_overlay, use_container_width=True)

    st.markdown(f"""
    <div style='background:{C['surface_soft']};border:1px solid {C['border_light']};
                border-radius:12px;padding:0.9rem 1.2rem;font-size:0.84rem;
                color:{C['text_muted']};margin-top:0.25rem;line-height:1.55;'>
      <b style='color:{C['text']};'>Cómo interpretarlo:</b>
      El rojo señala las áreas que más influyeron en la predicción de
      <i>{selected_name}</i>. Generado sobre el bloque convolucional denseblock4 de DenseNet-121.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    _render_results(probs, umbral, test_results)

    section_title("Información del modelo")
    model_cards(ckpt, test_results)

    section_title(
        "Exportar resultados",
        "Descarga un reporte PDF para presentación o un JSON con los datos crudos.",
    )

    col_pdf, col_json = st.columns(2)
    with col_pdf:
        with st.spinner("Generando PDF…"):
            pdf_bytes = build_pdf(
                img_pil, pil_overlay, probs, filename, ckpt or {}, test_results, umbral)
        st.download_button(
            "Descargar reporte PDF",
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
        gt = get_ground_truth(filename)
        if gt is not None:
            cmp = comparison or summarize_nih_comparison(probs, gt)
            export["etiqueta_nih"] = gt["labels_raw"]
            export["comparacion_nih"] = {
                "nih_positives": cmp["nih_positives"],
                "top_probabilidades": {
                    r["pathology"]: round(r["prob"], 4) for r in cmp["top_probs"]
                },
                "prob_elevada_sin_etiqueta": [
                    r["pathology"] for r in cmp["high_without_label"]
                ],
            }
        st.download_button(
            "Descargar JSON",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name=f"resultados_{Path(filename).stem}.json",
            mime="application/json",
            use_container_width=True,
        )


def render_sample_picker() -> Path | None:
    samples = list_sample_images()
    if not samples:
        return None
    st.markdown(
        f"<div style='font-size:0.78rem;color:{C['text_muted']};margin:0.4rem 0 0.25rem;"
        f"font-weight:600;text-transform:uppercase;letter-spacing:0.05em;'>"
        f"Ejemplos NIH</div>",
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