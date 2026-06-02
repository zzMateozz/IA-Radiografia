"""Componentes reutilizables de la interfaz — modernizados y sin emojis."""

from __future__ import annotations

import streamlit as st

from src.config import CHEXNET_AUC, FIGURES_DIR, PATHO_INFO, PATHOLOGIES
from ui.theme import PALETTE as C


def badge(text: str, kind: str = "neutral") -> str:
    colors = {
        "high":   (C["danger"],  "#F8D7DA"),
        "medium": (C["warning"], "#FFF3CD"),
        "low":    (C["text_light"], C["bg_alt"]),
        "success":(C["success"], "#D1E7DD"),
        "neutral":(C["text_muted"], C["bg_alt"]),
    }
    fg, bg = colors.get(kind, colors["neutral"])
    return (
        f"<span style='background:{bg};color:{fg};"
        f"border:1px solid {fg}33;border-radius:8px;"
        f"padding:2px 10px;font-size:0.72rem;font-weight:600;"
        f"letter-spacing:0.02em;display:inline-block;'>{text}</span>"
    )


def prob_bar(patho: str, prob: float, umbral: float, show_info: bool = True,
             test_results: dict | None = None) -> str:
    pct = int(prob * 100)
    if prob >= 0.5:
        bar_color, lbl_color, kind = C["danger"], C["danger"], "high"
        status = badge("Alta", "high")
    elif prob >= umbral:
        bar_color, lbl_color, kind = C["warning"], "#856404", "medium"
        status = badge("Media", "medium")
    else:
        bar_color, lbl_color = C["primary_light"], C["text_muted"]
        status = badge("Baja", "low")

    auc_hint = ""
    if show_info and prob >= umbral and test_results:
        our_auc = test_results.get("test_auc_per_class", {}).get(patho, 0)
        ref_auc = CHEXNET_AUC.get(patho, 0)
        auc_hint = (
            f"<div style='font-size:0.72rem;color:{C['text_light']};margin-top:4px;'>"
            f"AUC test {our_auc:.3f} · CheXNet {ref_auc:.3f}</div>"
        )

    info_html = (
        f"<div style='font-size:0.78rem;color:{C['text_muted']};margin-top:4px;line-height:1.4;'>"
        f"{PATHO_INFO[patho]}</div>{auc_hint}"
        if show_info and prob >= umbral else auc_hint
    )

    highlight = f"border-left:3px solid {bar_color};" if prob >= umbral else ""

    return f"""
    <div style='margin-bottom:0.85rem;background:{C['surface']};border:1px solid {C['border_light']};
                border-radius:12px;padding:0.85rem 1rem;box-shadow:{C['shadow_sm']};{highlight}'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
        <span style='font-size:0.92rem;font-weight:600;color:{lbl_color};'>{patho}</span>
        <div style='display:flex;align-items:center;gap:10px;'>
          {status}
          <span style='font-size:0.9rem;color:{lbl_color};font-weight:700;min-width:38px;text-align:right;'>
            {pct}%
          </span>
        </div>
      </div>
      <div style='background:{C['bg_alt']};border-radius:999px;height:8px;overflow:hidden;'>
        <div style='width:{pct}%;height:100%;background:{bar_color};border-radius:999px;
                    transition:width 0.4s ease;'></div>
      </div>
      {info_html}
    </div>"""


def card(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    accent = accent or C["primary"]
    sub_html = (
        f"<div style='color:{C['text_light']};font-size:0.76rem;margin-top:5px;line-height:1.35;'>"
        f"{sub}</div>"
        if sub else ""
    )
    return f"""
    <div style='background:{C['surface']};border:1px solid {C['border_light']};
                border-radius:12px;padding:1.1rem 1.25rem;box-shadow:{C['shadow_sm']};
                border-top:3px solid {accent};'>
      <div style='color:{C['text_light']};font-size:0.7rem;text-transform:uppercase;
                  letter-spacing:0.08em;font-weight:600;margin-bottom:6px;'>{label}</div>
      <div style='color:{C['text']};font-size:1.2rem;font-weight:700;line-height:1.2;'>{value}</div>
      {sub_html}
    </div>"""


def section_title(text: str, sub: str = "") -> None:
    st.markdown(
        f"<h3 style='margin:1.5rem 0 0.35rem;color:{C['text']};font-weight:600;font-size:1.25rem;'>{text}</h3>"
        + (f"<p style='color:{C['text_muted']};font-size:0.88rem;margin:0 0 1rem;"
           f"max-width:40rem;'>{sub}</p>" if sub else ""),
        unsafe_allow_html=True,
    )


def show_figure(fig_meta: dict) -> None:
    fig_path = FIGURES_DIR / fig_meta["file"]
    st.markdown(
        f"<p style='font-weight:600;color:{C['text']};margin-bottom:8px;font-size:0.95rem;'>"
        f"{fig_meta['title']}</p>",
        unsafe_allow_html=True,
    )
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info(f"Figura no encontrada: `{fig_path}`.")
    st.markdown(
        f"<p style='font-size:0.82rem;color:{C['text_muted']};margin-top:6px;"
        f"margin-bottom:1.5rem;line-height:1.5;'>{fig_meta['caption']}</p>",
        unsafe_allow_html=True,
    )


def hero_summary(probs, umbral: float) -> None:
    top_idx = int(probs.argmax())
    top_name = PATHOLOGIES[top_idx]
    top_prob = float(probs[top_idx])
    n_detected = sum(1 for p in probs if p >= umbral)
    n_alta = sum(1 for p in probs if p >= 0.5)

    if top_prob >= 0.5:
        level, color, kind = "Alta", C["danger"], "high"
    elif top_prob >= umbral:
        level, color, kind = "Media", C["warning"], "medium"
    else:
        level, color, kind = "Baja", C["primary"], "low"

    pct_bar = int(top_prob * 100)
    st.markdown(f"""
    <div class="hero-summary">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  flex-wrap:wrap;gap:1.2rem;">
        <div>
          <div style="color:{C['text_light']};font-size:0.72rem;text-transform:uppercase;
                      letter-spacing:0.1em;font-weight:600;">
            Resultado principal
          </div>
          <div style="color:{C['text']};font-size:1.55rem;font-weight:700;
                      margin:0.35rem 0 0.5rem;">
            {top_name}
          </div>
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            {badge(f"Confianza {level}", kind)}
            <span style="color:{color};font-size:1.15rem;font-weight:700;">{top_prob:.1%}</span>
          </div>
        </div>
        <div style="text-align:right;background:{C['surface_soft']};border-radius:14px;
                    padding:0.9rem 1.2rem;min-width:140px;">
          <div style="color:{C['text_light']};font-size:0.72rem;">Detectadas (umbral {umbral:.0%})</div>
          <div style="color:{C['primary']};font-size:2rem;font-weight:700;line-height:1.1;">
            {n_detected}
          </div>
          <div style="color:{C['text_muted']};font-size:0.75rem;">{n_alta} con confianza alta</div>
        </div>
      </div>
      <div style="background:{C['bg_alt']};border-radius:999px;height:10px;margin-top:1.2rem;overflow:hidden;">
        <div style="width:{pct_bar}%;height:100%;background:{color};border-radius:999px;"></div>
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
        st.metric("Test AUC", f"{auc_m:.4f}")
    with c3:
        st.metric("Mejor época", f"{best_ep} / 10")
    with c4:
        st.metric("Val AUC", f"{val_auc:.4f}")

    diff_color = C["success"] if auc_m >= chex_m else C["danger"]
    diff_txt = "por encima de CheXNet" if auc_m >= chex_m else "por debajo de CheXNet"
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:0.85rem;margin-top:1rem;'>
      {card("Dataset", "NIH ChestX-ray14", "112,120 radiografías · 30,805 pacientes", C['accent'])}
      {card("Entrenamiento", "10 épocas · T4", "Adam · BCE multi-etiqueta", C['info'])}
      {card("Referencia", f"CheXNet {chex_m:.3f}", f"Nuestro {auc_m:.3f} — {diff_txt}", diff_color)}
      {card("Técnica", "Transfer Learning", "ImageNet → dominio médico", C['primary'])}
    </div>""", unsafe_allow_html=True)