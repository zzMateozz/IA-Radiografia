"""Generación de reporte PDF con ReportLab."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import CHEXNET_AUC, PATHO_INFO, PATHOLOGIES
from src.inference.ground_truth import get_ground_truth, summarize_nih_comparison


def build_pdf(img_original, img_overlay, probs, filename, ckpt, test_results, umbral: float = 0.30):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )

    dark = colors.HexColor("#0f172a")
    gray = colors.HexColor("#64748b")
    light_bg = colors.HexColor("#f8fafc")

    styles = getSampleStyleSheet()
    title_st = ParagraphStyle("T", parent=styles["Title"],
                              fontName="Helvetica-Bold", fontSize=17,
                              textColor=dark, spaceAfter=3)
    sub_st = ParagraphStyle("S", parent=styles["Normal"],
                            fontName="Helvetica", fontSize=9,
                            textColor=gray, spaceAfter=2)
    h2_st = ParagraphStyle("H2", parent=styles["Heading2"],
                           fontName="Helvetica-Bold", fontSize=11,
                           textColor=dark, spaceBefore=12, spaceAfter=5)
    body_st = ParagraphStyle("B", parent=styles["Normal"],
                             fontName="Helvetica", fontSize=8,
                             textColor=dark, spaceAfter=3)
    warn_st = ParagraphStyle("W", parent=styles["Normal"],
                             fontName="Helvetica-Oblique", fontSize=8,
                             textColor=gray)

    story = []
    story.append(Paragraph("Reporte de Clasificación de Radiografía de Tórax", title_st))
    story.append(Paragraph(
        "Universidad Popular del Cesar · Inteligencia Artificial 2026-I", sub_st))
    story.append(Paragraph(
        "Integrantes: Mateo Lopez Patiño · Anaclaudia Vega Martinez", sub_st))
    story.append(Paragraph(
        "Docente: Tonny Enrique Jimenez Marquez", sub_st))
    story.append(Paragraph(
        f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}  ·  Archivo: {filename}", sub_st))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "AVISO: Reporte de uso exclusivamente académico. "
        "No reemplaza el criterio de un médico radiólogo especializado.", warn_st))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Análisis de imagen", h2_st))

    def pil_to_rl(pil_img, w):
        tmp = io.BytesIO()
        pil_img.convert("RGB").resize((256, 256)).save(tmp, format="PNG")
        tmp.seek(0)
        return RLImage(tmp, width=w * inch, height=w * inch)

    lbl_st = ParagraphStyle("lbl", parent=styles["Normal"],
                            fontName="Helvetica-Bold", fontSize=8,
                            textColor=dark, alignment=1)
    img_tbl = Table(
        [[Paragraph("Radiografía original", lbl_st),
          Paragraph("Grad-CAM (zona de atención)", lbl_st)],
         [pil_to_rl(img_original, 2.7), pil_to_rl(img_overlay, 2.7)]],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    img_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), light_bg),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 12))

    gt = get_ground_truth(filename)
    if gt is not None:
        cmp = summarize_nih_comparison(probs, gt)
        nih_text = gt["labels_raw"] if not gt["no_finding"] else "No Finding"
        top_text = ", ".join(
            f"{r['pathology']} ({r['prob']:.1%})" for r in cmp["top_probs"]
        )

        story.append(Paragraph("Comparación con etiqueta NIH", h2_st))
        cmp_rows = [
            [Paragraph("<b>Etiqueta NIH</b>", body_st), Paragraph(nih_text, body_st)],
            [Paragraph("<b>Mayores probabilidades</b>", body_st), Paragraph(top_text, body_st)],
        ]
        if cmp["high_without_label"]:
            extra = ", ".join(r["pathology"] for r in cmp["high_without_label"])
            cmp_rows.append([
                Paragraph("<b>Prob. ≥ 50% sin etiqueta NIH</b>", body_st),
                Paragraph(extra, body_st),
            ])
        cmp_tbl = Table(cmp_rows, colWidths=[1.7 * inch, 5.0 * inch])
        cmp_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), dark),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, light_bg]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(cmp_tbl)
        story.append(Spacer(1, 12))

    story.append(Paragraph("Resultados por patología", h2_st))
    sorted_idx = np.argsort(probs)[::-1]
    header = [Paragraph(f"<b>{t}</b>", body_st)
              for t in ["Patología", "Probabilidad", "Estado", "Descripción"]]
    rows = [header]
    for i in sorted_idx:
        p, prob = PATHOLOGIES[i], probs[i]
        if prob >= 0.5:
            estado = Paragraph(f'<font color="#dc2626"><b>ALTA ({prob:.1%})</b></font>', body_st)
        elif prob >= umbral:
            estado = Paragraph(f'<font color="#ea580c"><b>MEDIA ({prob:.1%})</b></font>', body_st)
        else:
            estado = Paragraph(f'<font color="#64748b">Baja ({prob:.1%})</font>', body_st)
        rows.append([
            Paragraph(f"<b>{p}</b>" if prob >= umbral else p, body_st),
            Paragraph(f"{prob:.1%}", body_st),
            estado,
            Paragraph(PATHO_INFO[p], body_st),
        ])
    pt = Table(rows, colWidths=[1.3 * inch, 0.85 * inch, 1.1 * inch, 3.45 * inch])
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), light_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), dark),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), dark),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    for ri, i in enumerate(sorted_idx, 1):
        if probs[i] >= 0.5:
            ts.add("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#fef2f2"))
        elif probs[i] >= umbral:
            ts.add("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#fff7ed"))
    pt.setStyle(ts)
    story.append(pt)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Información del modelo", h2_st))
    auc_m = test_results["test_auc_mean"] if test_results else 0.0
    best_ep = test_results["best_epoch"] if test_results else (ckpt or {}).get("epoch", "?")
    val_auc = test_results["best_val_auc"] if test_results else (ckpt or {}).get("val_auc", 0.0)
    info_rows = [
        ["Arquitectura", "DenseNet-121 con Transfer Learning (ImageNet → NIH ChestX-ray14)"],
        ["Dataset", "NIH ChestX-ray14 — 112,120 radiografías · 30,805 pacientes únicos"],
        ["Entrenamiento", "10 épocas · GPU Tesla T4 · Adam lr=1e-4 · BCE multi-etiqueta · batch=32"],
        ["Mejor época", f"Época {best_ep} / 10 — Val AUC: {val_auc:.4f}"],
        ["Test AUC medio", "%.4f  (CheXNet Stanford ref.: %.4f)" % (auc_m, sum(CHEXNET_AUC.values()) / 14)],
        ["Umbral detección", f"{umbral:.0%} de probabilidad para marcar hallazgo"],
    ]
    it = Table(info_rows, colWidths=[1.7 * inch, 5.0 * inch])
    it.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), dark),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, light_bg]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(it)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Este reporte fue generado automáticamente. "
        "No constituye diagnóstico médico. Consulte siempre a un especialista.", warn_st))

    doc.build(story)
    buf.seek(0)
    return buf.read()
