"""Componentes reutilizables de la interfaz Streamlit."""

from __future__ import annotations

import streamlit as st

from src.config import CHEXNET_AUC, FIGURES_DIR, PATHO_INFO, PATHOLOGIES


def badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}22;color:{color};"
        f"border:1px solid {color}44;border-radius:999px;"
        f"padding:2px 10px;font-size:0.75rem;font-weight:600;"
        f"font-family:\"JetBrains Mono\",monospace;'>{text}</span>"
    )


def prob_bar(patho: str, prob: float, umbral: float, show_info: bool = True,
             test_results: dict | None = None) -> str:
    pct = int(prob * 100)
    if prob >= 0.5:
        bar_color = lbl_color = "#ef4444"
        status = badge("ALTA", "#ef4444")
    elif prob >= umbral:
        bar_color = lbl_color = "#f97316"
        status = badge("MEDIA", "#f97316")
    else:
        bar_color = "#1d4ed8"
        lbl_color = "#475569"
        status = badge("BAJA", "#475569")

    auc_hint = ""
    if show_info and prob >= umbral and test_results:
        our_auc = test_results.get("test_auc_per_class", {}).get(patho, 0)
        ref_auc = CHEXNET_AUC.get(patho, 0)
        auc_hint = (
            f"<div style='font-size:0.7rem;color:#64748b;margin-top:2px;'>"
            f"AUC test: {our_auc:.3f} · CheXNet ref.: {ref_auc:.3f}</div>"
        )

    info_html = (
        f"<div style='font-size:0.73rem;color:#4d6a99;margin-top:2px;'>"
        f"{PATHO_INFO[patho]}</div>{auc_hint}"
        if show_info and prob >= umbral else auc_hint
    )
    return f"""
    <div style='margin-bottom:0.75rem;'>
      <div style='display:flex;justify-content:space-between;
                  align-items:center;margin-bottom:4px;'>
        <span style='font-size:0.9rem;font-weight:600;color:{lbl_color};'>{patho}</span>
        <div style='display:flex;align-items:center;gap:8px;'>
          {status}
          <span style='font-family:"JetBrains Mono",monospace;font-size:0.85rem;
                       color:{lbl_color};font-weight:700;min-width:40px;
                       text-align:right;'>{pct}%</span>
        </div>
      </div>
      <div style='background:#0d1f3c;border-radius:999px;height:7px;overflow:hidden;'>
        <div style='width:{pct}%;height:100%;
                    background:linear-gradient(90deg,{bar_color}88,{bar_color});
                    border-radius:999px;'></div>
      </div>
      {info_html}
    </div>"""


def card(label: str, value: str, sub: str = "") -> str:
    sub_html = (
        f"<div style='color:#4d6a99;font-size:0.75rem;margin-top:3px;'>{sub}</div>"
        if sub else ""
    )
    return f"""
    <div style='background:#0d1f3c;border:1px solid #1a3a6b;border-radius:12px;
                padding:1rem 1.2rem;'>
      <div style='color:#4d6a99;font-size:0.72rem;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:4px;'>{label}</div>
      <div style='color:#60a5fa;font-family:"JetBrains Mono",monospace;
                  font-size:1.25rem;font-weight:600;'>{value}</div>
      {sub_html}
    </div>"""


def section_title(icon: str, text: str, sub: str = "") -> None:
    html = f"<h3 style='margin:0 0 0.2rem;color:#f1f5f9;'>{icon} {text}</h3>"
    if sub:
        html += f"<p style='color:#4d6a99;font-size:0.85rem;margin:0 0 1rem;'>{sub}</p>"
    st.markdown(html, unsafe_allow_html=True)


def show_figure(fig_meta: dict) -> None:
    fig_path = FIGURES_DIR / fig_meta["file"]
    st.markdown(
        f"<p style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>{fig_meta['title']}</p>",
        unsafe_allow_html=True,
    )
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info(f"Figura no encontrada: `{fig_path}`.")
    st.markdown(
        f"<p style='font-size:0.8rem;color:#4d6a99;margin-top:4px;"
        f"margin-bottom:1.5rem;'>{fig_meta['caption']}</p>",
        unsafe_allow_html=True,
    )


def hero_summary(probs, umbral: float) -> None:
    """Tarjeta resumen del hallazgo principal (Fase 1)."""
    top_idx = int(probs.argmax())
    top_name = PATHOLOGIES[top_idx]
    top_prob = float(probs[top_idx])
    n_detected = sum(1 for p in probs if p >= umbral)
    n_alta = sum(1 for p in probs if p >= 0.5)

    if top_prob >= 0.5:
        level, color = "ALTA", "#ef4444"
    elif top_prob >= umbral:
        level, color = "MEDIA", "#f97316"
    else:
        level, color = "BAJA", "#60a5fa"

    pct_bar = int(top_prob * 100)
    st.markdown(f"""
    <div class="hero-summary">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
        <div>
          <div style="color:#4d6a99;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">
            Hallazgo principal
          </div>
          <div style="color:#f1f5f9;font-size:1.4rem;font-weight:700;margin:0.3rem 0;">
            {top_name}
          </div>
          <div style="color:{color};font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:600;">
            {top_prob:.1%} · confianza {level}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="color:#4d6a99;font-size:0.75rem;">Patologías ≥ {umbral:.0%}</div>
          <div style="color:#60a5fa;font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;">
            {n_detected}
          </div>
          <div style="color:#4d6a99;font-size:0.72rem;">{n_alta} con confianza alta (≥50%)</div>
        </div>
      </div>
      <div style="background:#060d1a;border-radius:999px;height:8px;margin-top:1rem;overflow:hidden;">
        <div style="width:{pct_bar}%;height:100%;background:linear-gradient(90deg,{color}88,{color});border-radius:999px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def model_cards(ckpt, test_results) -> None:
    auc_m = test_results["test_auc_mean"] if test_results else 0.0
    best_ep = test_results["best_epoch"] if test_results else (ckpt or {}).get("epoch", "?")
    val_auc = test_results["best_val_auc"] if test_results else (ckpt or {}).get("val_auc", 0.0)
    chex_m = sum(CHEXNET_AUC.values()) / len(CHEXNET_AUC)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Arquitectura", "DenseNet-121")
    with c2:
        st.metric("Test AUC promedio", f"{auc_m:.4f}")
    with c3:
        st.metric("Mejor época", f"{best_ep} / 10")
    with c4:
        st.metric("Val AUC (mejor época)", f"{val_auc:.4f}")

    diff_str = (
        '<span style="color:#4ade80">▲ por encima</span>'
        if auc_m >= chex_m else
        '<span style="color:#f97316">▼ por debajo</span>'
    )
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;margin-top:0.8rem;'>
      {card("DATASET", "NIH ChestX-ray14", "112,120 radiografías · 30,805 pacientes")}
      {card("ENTRENAMIENTO", "10 épocas · Tesla T4", "Adam lr=1e-4 · BCE multi-etiqueta")}
      {card("REFERENCIA", f"CheXNet {chex_m:.4f}", f"Nuestro: {auc_m:.4f} {diff_str}")}
      {card("TÉCNICA", "Transfer Learning", "ImageNet preentrenado → dominio médico")}
    </div>""", unsafe_allow_html=True)
