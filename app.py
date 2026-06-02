"""
app.py — Sistema de Clasificación de Radiografías de Tórax
Universidad Popular del Cesar — Inteligencia Artificial 2026-I
Mateo Lopez Patiño · Anaclaudia Vega Martinez · Tonny Enrique Jimenez Marquez

Ejecutar:
    streamlit run app.py
"""

import io
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  RUTAS
# ══════════════════════════════════════════════════════════
MODELS_DIR  = Path("outputs/models")
FIGURES_DIR = Path("outputs/figures")

# ══════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

CHEXNET_AUC = {
    "Atelectasis": 0.8094, "Cardiomegaly": 0.9248, "Effusion": 0.8638,
    "Infiltration": 0.7345, "Mass": 0.8676, "Nodule": 0.7802,
    "Pneumonia": 0.7680, "Pneumothorax": 0.8887, "Consolidation": 0.7901,
    "Edema": 0.8878, "Emphysema": 0.9371, "Fibrosis": 0.8047,
    "Pleural_Thickening": 0.7856, "Hernia": 0.9164,
}

PATHO_INFO = {
    "Atelectasis":        "Colapso parcial o total de un lóbulo pulmonar.",
    "Cardiomegaly":       "Agrandamiento del corazón visible en la radiografía.",
    "Effusion":           "Acumulación de líquido en el espacio pleural.",
    "Infiltration":       "Opacidades difusas por líquido, pus o células en el pulmón.",
    "Mass":               "Lesión sólida mayor de 3 cm de diámetro.",
    "Nodule":             "Lesión redondeada menor de 3 cm, puede ser benigna o maligna.",
    "Pneumonia":          "Infección que inflama los sacos de aire del pulmón.",
    "Pneumothorax":       "Presencia de aire en el espacio pleural.",
    "Consolidation":      "Tejido pulmonar reemplazado por fluido o tejido sólido.",
    "Edema":              "Acumulación de líquido en el tejido pulmonar.",
    "Emphysema":          "Destrucción de los alvéolos, común en fumadores.",
    "Fibrosis":           "Cicatrización y engrosamiento del tejido pulmonar.",
    "Pleural_Thickening": "Engrosamiento de la pleura, puede restringir la respiración.",
    "Hernia":             "Protrusión de órganos abdominales hacia el tórax.",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
UMBRAL        = 0.30

# Figuras EDA con título y explicación
EDA_FIGURES = [
    {
        "file":    "01_distribucion_patologias.png",
        "title":   "Distribución de las 14 patologías",
        "caption": (
            "Infiltration es la patología más frecuente (17.7 %), seguida de Effusion (11.9 %) "
            "y Atelectasis (10.3 %). Hernia es la más rara con apenas 227 imágenes (0.2 %). "
            "Este desbalance extremo es uno de los principales retos del entrenamiento."
        ),
    },
    {
        "file":    "02_no_finding_vs_finding.png",
        "title":   "'No Finding' vs hallazgo positivo",
        "caption": (
            "El 53.8 % de las imágenes (60,361) no presentan ninguna patología etiquetada. "
            "El 46.2 % restante (51,759) tiene al menos un hallazgo. Este desbalance entre "
            "clases normales y patológicas justifica el uso de pesos en la función de pérdida."
        ),
    },
    {
        "file":    "03_hallazgos_por_imagen.png",
        "title":   "Cantidad de hallazgos por radiografía",
        "caption": (
            "La mayoría de imágenes positivas tiene 1 o 2 hallazgos simultáneos. "
            "Algunas llegan a tener hasta 9 patologías en una sola radiografía, "
            "lo que convierte esto en un problema de clasificación multi-etiqueta real."
        ),
    },
    {
        "file":    "04_cooccurrence.png",
        "title":   "Co-ocurrencia condicional de patologías",
        "caption": (
            "Cada celda muestra P(columna | fila): la probabilidad de encontrar una patología "
            "dado que ya existe otra. Edema e Infiltration co-ocurren con frecuencia (43 %). "
            "Cardiomegaly aparece junto a Effusion en el 38 % de los casos. "
            "Estas correlaciones sugieren que el modelo puede aprender señales compartidas."
        ),
    },
    {
        "file":    "05_demografia.png",
        "title":   "Demografía del dataset",
        "caption": (
            "Mediana de edad: 49 años, con distribución bimodal alrededor de los 40-60 años. "
            "El 56.5 % de las imágenes corresponde a pacientes masculinos. "
            "El 60 % de las radiografías es en vista PA (posteroanterior), "
            "el 40 % en AP (anteroposterior, típicamente pacientes encamados)."
        ),
    },
    {
        "file":    "06_prevalencia_por_edad.png",
        "title":   "Prevalencia por grupo de edad",
        "caption": (
            "Infiltration es consistentemente la patología más prevalente en todos los grupos. "
            "Atelectasis y Effusion aumentan con la edad, alcanzando 13.1 % y 17.2 % en mayores "
            "de 76 años respectivamente. Hernia es prácticamente inexistente en menores de 45."
        ),
    },
    {
        "file":    "07_imgs_por_paciente.png",
        "title":   "Distribución de imágenes por paciente",
        "caption": (
            "La mayoría de los 30,805 pacientes tiene entre 1 y 5 radiografías. "
            "Sin embargo, algunos pacientes tienen más de 100 imágenes (escala logarítmica). "
            "La división train/val/test se hizo por paciente para evitar data leakage."
        ),
    },
]

# Figuras de evaluación generadas por el notebook 03
EVAL_FIGURES = [
    {
        "file":    "08_curvas_aprendizaje.png",
        "title":   "Curvas de aprendizaje — Pérdida y AUC por época",
        "caption": (
            "La pérdida de entrenamiento baja consistentemente en las 10 épocas. "
            "La pérdida de validación se estabiliza alrededor de la época 5 y sube levemente "
            "después (señal de overfitting suave). El mejor AUC de validación (0.8178) "
            "se alcanzó en la época 7, que es el checkpoint guardado como best_model.pt."
        ),
    },
    {
        "file":    "09_auc_por_clase.png",
        "title":   "AUC-ROC por patología — Nuestro modelo vs CheXNet",
        "caption": (
            "Hernia (0.912), Emphysema (0.886) y Cardiomegaly (0.888) son las clases "
            "con mejor AUC. Pneumonia (0.677) e Infiltration (0.688) son las más difíciles, "
            "lo que concuerda con la literatura: son patologías con presentación visual difusa "
            "y alta variabilidad entre pacientes."
        ),
    },
    {
        "file":    "10_curvas_roc.png",
        "title":   "Curvas ROC — una por cada patología",
        "caption": (
            "Cada curva muestra la relación entre tasa de verdaderos positivos (TPR) "
            "y tasa de falsos positivos (FPR) para cada umbral de decisión. "
            "Las curvas más alejadas de la diagonal son las que el modelo clasifica mejor. "
            "La línea diagonal punteada representa clasificación aleatoria (AUC=0.5)."
        ),
    },
    {
        "file":    "11_distribucion_predicciones.png",
        "title":   "Distribución de probabilidades — Positivos vs Negativos",
        "caption": (
            "Para cada patología se muestran las probabilidades que asigna el modelo "
            "a casos realmente positivos (rojo) vs negativos (azul). "
            "Una buena separación entre ambas distribuciones indica alto poder discriminativo. "
            "Hernia y Emphysema muestran mejor separación que Pneumonia o Consolidation."
        ),
    },
]


# ══════════════════════════════════════════════════════════
#  MODELO
# ══════════════════════════════════════════════════════════
class DenseNetCheXNet(nn.Module):
    def __init__(self, n_classes: int = 14):
        super().__init__()
        backbone = models.densenet121(weights=None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, n_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


@st.cache_resource(show_spinner="Cargando modelo DenseNet-121…")
def load_model():
    path = MODELS_DIR / "best_model.pt"
    if not path.exists():
        return None, None
    model = DenseNetCheXNet()
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data(show_spinner=False)
def load_test_results():
    path = MODELS_DIR / "test_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════
#  INFERENCIA
# ══════════════════════════════════════════════════════════
def get_transform():
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict(model, img_pil: Image.Image):
    tensor = get_transform()(img_pil)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor.unsqueeze(0))).squeeze().numpy()
    return probs, tensor


# ══════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════
def generate_gradcam(model, img_tensor: torch.Tensor, target_idx: int) -> np.ndarray:
    target_layer  = model.backbone.features.denseblock4
    gradients, activations = [], []

    def save_grad(grad):
        gradients.append(grad)

    def fwd_hook(module, inp, out):
        activations.append(out)
        out.register_hook(save_grad)

    hook   = target_layer.register_forward_hook(fwd_hook)
    inp    = img_tensor.unsqueeze(0).requires_grad_(True)
    logits = model(inp)
    model.zero_grad()
    logits[0, target_idx].backward()
    hook.remove()

    grads   = gradients[0].squeeze().detach().numpy()
    acts    = activations[0].squeeze().detach().numpy()
    weights = grads.mean(axis=(1, 2))
    cam     = np.maximum((weights[:, None, None] * acts).sum(axis=0), 0)
    cam    -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def make_gradcam_overlay(model, img_pil, img_tensor, target_idx):
    cam         = generate_gradcam(model, img_tensor, target_idx)
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224)))
    img_rgb     = denormalize(img_tensor)
    heatmap     = mpl_cm.jet(cam_resized)[:, :, :3]
    overlay     = np.clip(0.55 * img_rgb + 0.45 * heatmap, 0, 1)
    return img_rgb, cam_resized, overlay


def ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
#  GENERACIÓN DE PDF
# ══════════════════════════════════════════════════════════
def build_pdf(img_original, img_overlay, probs, filename, ckpt, test_results):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
    )

    DARK  = colors.HexColor("#0f172a")
    BLUE  = colors.HexColor("#1d4ed8")
    GRAY  = colors.HexColor("#64748b")

    styles   = getSampleStyleSheet()
    title_st = ParagraphStyle("T", parent=styles["Title"],
                              fontName="Helvetica-Bold", fontSize=17,
                              textColor=DARK, spaceAfter=3)
    sub_st   = ParagraphStyle("S", parent=styles["Normal"],
                              fontName="Helvetica", fontSize=9,
                              textColor=GRAY, spaceAfter=2)
    h2_st    = ParagraphStyle("H2", parent=styles["Heading2"],
                              fontName="Helvetica-Bold", fontSize=11,
                              textColor=BLUE, spaceBefore=12, spaceAfter=5)
    body_st  = ParagraphStyle("B", parent=styles["Normal"],
                              fontName="Helvetica", fontSize=8,
                              textColor=DARK, spaceAfter=3)
    warn_st  = ParagraphStyle("W", parent=styles["Normal"],
                              fontName="Helvetica-Oblique", fontSize=8,
                              textColor=GRAY)

    story = []

    # Encabezado
    story.append(Paragraph("Reporte de Clasificación de Radiografía de Tórax", title_st))
    story.append(Paragraph(
        "Universidad Popular del Cesar · Inteligencia Artificial 2026-I", sub_st))
    story.append(Paragraph(
        "Mateo Lopez Patiño · Anaclaudia Vega Martinez · Tonny Enrique Jimenez Marquez", sub_st))
    story.append(Paragraph(
        f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}  ·  Archivo: {filename}", sub_st))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=8))
    story.append(Paragraph(
        "AVISO: Reporte de uso exclusivamente académico. "
        "No reemplaza el criterio de un médico radiólogo especializado.", warn_st))
    story.append(Spacer(1, 10))

    # Imágenes
    story.append(Paragraph("Análisis de imagen", h2_st))

    def pil_to_rl(pil_img, w):
        tmp = io.BytesIO()
        pil_img.convert("RGB").resize((256, 256)).save(tmp, format="PNG")
        tmp.seek(0)
        return RLImage(tmp, width=w*inch, height=w*inch)

    lbl_st = ParagraphStyle("lbl", parent=styles["Normal"],
                             fontName="Helvetica-Bold", fontSize=8,
                             textColor=DARK, alignment=1)
    img_tbl = Table(
        [[Paragraph("Radiografía original", lbl_st),
          Paragraph("Grad-CAM (zona de atención)", lbl_st)],
         [pil_to_rl(img_original, 2.7), pil_to_rl(img_overlay, 2.7)]],
        colWidths=[3.2*inch, 3.2*inch],
    )
    img_tbl.setStyle(TableStyle([
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("BACKGROUND",   (0,0), (-1, 0), colors.HexColor("#f8fafc")),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 12))

    # Tabla de patologías
    story.append(Paragraph("Resultados por patología", h2_st))
    sorted_idx = np.argsort(probs)[::-1]
    header = [Paragraph(f"<b>{t}</b>", body_st)
              for t in ["Patología", "Probabilidad", "Estado", "Descripción"]]
    rows = [header]
    for i in sorted_idx:
        p, prob = PATHOLOGIES[i], probs[i]
        if prob >= 0.5:
            estado = Paragraph(f'<font color="#dc2626"><b>ALTA ({prob:.1%})</b></font>', body_st)
        elif prob >= UMBRAL:
            estado = Paragraph(f'<font color="#ea580c"><b>MEDIA ({prob:.1%})</b></font>', body_st)
        else:
            estado = Paragraph(f'<font color="#64748b">Baja ({prob:.1%})</font>', body_st)
        rows.append([
            Paragraph(f"<b>{p}</b>" if prob >= UMBRAL else p, body_st),
            Paragraph(f"{prob:.1%}", body_st),
            estado,
            Paragraph(PATHO_INFO[p], body_st),
        ])
    pt = Table(rows, colWidths=[1.3*inch, 0.85*inch, 1.1*inch, 3.45*inch])
    ts = TableStyle([
        ("BACKGROUND",   (0,0), (-1, 0), BLUE),
        ("TEXTCOLOR",    (0,0), (-1, 0), colors.white),
        ("FONTNAME",     (0,0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ])
    for ri, i in enumerate(sorted_idx, 1):
        if probs[i] >= 0.5:
            ts.add("BACKGROUND", (0,ri), (-1,ri), colors.HexColor("#fef2f2"))
        elif probs[i] >= UMBRAL:
            ts.add("BACKGROUND", (0,ri), (-1,ri), colors.HexColor("#fff7ed"))
    pt.setStyle(ts)
    story.append(pt)
    story.append(Spacer(1, 12))

    # Info del modelo
    story.append(HRFlowable(width="100%", thickness=0.4,
                            color=colors.HexColor("#e2e8f0"), spaceAfter=6))
    story.append(Paragraph("Información del modelo", h2_st))
    auc_m   = test_results["test_auc_mean"] if test_results else 0.0
    best_ep = test_results["best_epoch"]    if test_results else (ckpt or {}).get("epoch","?")
    val_auc = test_results["best_val_auc"]  if test_results else (ckpt or {}).get("val_auc", 0.0)
    info_rows = [
        ["Arquitectura",     "DenseNet-121 con Transfer Learning (ImageNet → NIH ChestX-ray14)"],
        ["Dataset",          "NIH ChestX-ray14 — 112,120 radiografías · 30,805 pacientes únicos"],
        ["Entrenamiento",    "10 épocas · GPU Tesla T4 · Adam lr=1e-4 · BCE multi-etiqueta · batch=32"],
        ["Mejor época",      f"Época {best_ep} / 10 — Val AUC: {val_auc:.4f}"],
        ["Test AUC medio",   "%.4f  (CheXNet Stanford ref.: %.4f)" % (auc_m, sum(CHEXNET_AUC.values())/14)],
        ["Umbral detección", f"{UMBRAL:.0%} de probabilidad para marcar hallazgo"],
    ]
    it = Table(info_rows, colWidths=[1.7*inch, 5.0*inch])
    it.setStyle(TableStyle([
        ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (1,0),(1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("TEXTCOLOR",     (0,0),(0,-1), BLUE),
        ("TEXTCOLOR",     (1,0),(1,-1), DARK),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    story.append(it)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Este reporte fue generado automáticamente. "
        "No constituye diagnóstico médico. Consulte siempre a un especialista.", warn_st))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
#  CSS GLOBAL
# ══════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #060d1a;
    color: #e2e8f0;
}
.stApp { background: #060d1a; }

section[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1a2640;
}
h1,h2,h3,h4 { color:#f1f5f9 !important; font-family:'Space Grotesk',sans-serif !important; }

[data-testid="metric-container"] {
    background: #0d1f3c;
    border: 1px solid #1a3a6b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-family:'JetBrains Mono',monospace !important;
    color:#60a5fa !important; font-size:1.5rem !important; font-weight:600 !important;
}
[data-testid="stMetricLabel"] { color:#4d6a99 !important; font-size:0.78rem !important; }

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg,#1d4ed8,#3b82f6) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important;
    transition: all 0.2s !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    opacity:0.88 !important; transform:translateY(-1px);
}
[data-testid="stFileUploaderDropzone"] {
    background:#0d1f3c !important;
    border:2px dashed #1a3a6b !important;
    border-radius:14px !important;
}
hr { border-color:#1a2640 !important; }
[data-testid="stExpander"] {
    background:#0d1f3c; border:1px solid #1a3a6b !important; border-radius:10px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background:#0a1628; border-radius:10px; padding:4px; gap:4px;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#4d6a99 !important;
    border-radius:8px !important; font-weight:500;
}
.stTabs [aria-selected="true"] {
    background:#1d4ed8 !important; color:white !important;
}
</style>
"""


# ══════════════════════════════════════════════════════════
#  HELPERS UI
# ══════════════════════════════════════════════════════════
def badge(text, color):
    return (
        f"<span style='background:{color}22;color:{color};"
        f"border:1px solid {color}44;border-radius:999px;"
        f"padding:2px 10px;font-size:0.75rem;font-weight:600;"
        f"font-family:\"JetBrains Mono\",monospace;'>{text}</span>"
    )


def prob_bar(patho, prob, show_info=True):
    pct = int(prob * 100)
    if prob >= 0.5:
        bar_color = lbl_color = "#ef4444"
        status = badge("ALTA", "#ef4444")
    elif prob >= UMBRAL:
        bar_color = lbl_color = "#f97316"
        status = badge("MEDIA", "#f97316")
    else:
        bar_color = "#1d4ed8"; lbl_color = "#475569"
        status = badge("BAJA", "#475569")

    info_html = (
        f"<div style='font-size:0.73rem;color:#4d6a99;margin-top:2px;'>"
        f"{PATHO_INFO[patho]}</div>"
        if show_info and prob >= UMBRAL else ""
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


def card(label, value, sub=""):
    return f"""
    <div style='background:#0d1f3c;border:1px solid #1a3a6b;border-radius:12px;
                padding:1rem 1.2rem;'>
      <div style='color:#4d6a99;font-size:0.72rem;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:4px;'>{label}</div>
      <div style='color:#60a5fa;font-family:"JetBrains Mono",monospace;
                  font-size:1.25rem;font-weight:600;'>{value}</div>
      {"<div style='color:#4d6a99;font-size:0.75rem;margin-top:3px;'>"+sub+"</div>" if sub else ""}
    </div>"""


def section_title(icon, text, sub=""):
    s = f"<h3 style='margin:0 0 0.2rem;color:#f1f5f9;'>{icon} {text}</h3>"
    if sub:
        s += f"<p style='color:#4d6a99;font-size:0.85rem;margin:0 0 1rem;'>{sub}</p>"
    st.markdown(s, unsafe_allow_html=True)


def show_figure(fig_meta, key_prefix=""):
    """Muestra una figura desde outputs/figures/ con su caption."""
    fig_path = FIGURES_DIR / fig_meta["file"]
    st.markdown(
        f"<p style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
        f"{fig_meta['title']}</p>",
        unsafe_allow_html=True,
    )
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info(
            f"Figura no encontrada: `{fig_path}`. "
            "Ejecuta primero el notebook `01_eda_metadata.ipynb`."
        )
    st.markdown(
        f"<p style='font-size:0.8rem;color:#4d6a99;margin-top:4px;"
        f"margin-bottom:1.5rem;'>{fig_meta['caption']}</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
#  SECCIÓN: ANÁLISIS DE IMAGEN
# ══════════════════════════════════════════════════════════
def section_analisis(model, ckpt, test_results):
    section_title(
        "🔬", "Análisis de imagen",
        "Sube una radiografía de tórax (PNG · JPG) para obtener predicción "
        "de 14 patologías y mapa de calor Grad-CAM.",
    )

    uploaded = st.file_uploader(
        "Radiografía", type=["png","jpg","jpeg"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div style='border:2px dashed #1a3a6b;border-radius:16px;padding:4rem;
                    text-align:center;color:#4d6a99;background:#0d1f3c;margin-top:1rem;'>
          <div style='font-size:3.5rem;margin-bottom:0.5rem;'>🫁</div>
          <div style='font-size:1.05rem;font-weight:600;color:#60a5fa;'>
            Arrastra o selecciona una radiografía
          </div>
          <div style='font-size:0.82rem;margin-top:0.4rem;'>PNG · JPG · JPEG</div>
        </div>""", unsafe_allow_html=True)
        return

    img_pil = Image.open(uploaded).convert("RGB")
    probs, img_tensor = predict(model, img_pil)

    # Selector de patología para Grad-CAM
    top_idx = int(probs.argmax())
    with st.expander("🎯 Seleccionar patología para Grad-CAM", expanded=False):
        selected_name = st.radio(
            "p", PATHOLOGIES, index=top_idx,
            horizontal=True, label_visibility="collapsed",
        )
    target_idx = PATHOLOGIES.index(selected_name)

    with st.spinner("Generando Grad-CAM…"):
        img_rgb, cam_arr, overlay_arr = make_gradcam_overlay(
            model, img_pil, img_tensor, target_idx)
    pil_orig    = ndarray_to_pil(img_rgb)
    pil_overlay = ndarray_to_pil(overlay_arr)

    # ── Grad-CAM: dos imágenes lado a lado
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
            unsafe_allow_html=True)
        st.image(pil_orig, use_container_width=True)
    with col_cam:
        st.markdown(
            f"<p style='text-align:center;color:#4d6a99;font-size:0.85rem;"
            f"margin-bottom:6px;'>🌡 Grad-CAM · {selected_name}</p>",
            unsafe_allow_html=True)
        st.image(pil_overlay, use_container_width=True)

    st.markdown("""
    <div style='background:#0d1f3c;border:1px solid #1a3a6b;border-radius:10px;
                padding:0.8rem 1.2rem;font-size:0.8rem;color:#4d6a99;margin-top:0.5rem;'>
      <b style='color:#e2e8f0;'>¿Cómo leer el mapa?</b> &nbsp;
      Zonas en <span style='color:#ef4444;'>■ rojo</span> = región más relevante
      para la predicción. Zonas en <span style='color:#3b82f6;'>■ azul</span> = baja
      relevancia. Generado con Grad-CAM sobre el último bloque convolucional de DenseNet-121.
    </div>""", unsafe_allow_html=True)

    # ── Resultados de las 14 patologías
    st.markdown("---")
    section_title("📋", "Resultados — 14 patologías")

    positivos = sorted(
        [(p, probs[i]) for i,p in enumerate(PATHOLOGIES) if probs[i] >= UMBRAL],
        key=lambda x: x[1], reverse=True)
    negativos = sorted(
        [(p, probs[i]) for i,p in enumerate(PATHOLOGIES) if probs[i] < UMBRAL],
        key=lambda x: x[1], reverse=True)

    col_pos, col_neg = st.columns([1.1, 1], gap="large")
    with col_pos:
        if positivos:
            n_a = sum(1 for _,v in positivos if v >= 0.5)
            n_m = sum(1 for _,v in positivos if UMBRAL <= v < 0.5)
            st.markdown(
                f"<div style='margin-bottom:1rem;'>"
                f"{badge(f'{n_a} alta(s)', '#ef4444')} &nbsp;"
                f"{badge(f'{n_m} media(s)', '#f97316')}"
                f"</div>",
                unsafe_allow_html=True)
            st.markdown("".join(prob_bar(p,v) for p,v in positivos),
                        unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#052e16;border:1px solid #166534;
                        border-radius:12px;padding:1.5rem;text-align:center;'>
              <div style='font-size:1.8rem;'>✅</div>
              <div style='color:#4ade80;font-weight:600;margin-top:0.4rem;'>
                Sin hallazgos detectados
              </div>
              <div style='color:#166534;font-size:0.82rem;margin-top:0.3rem;'>
                Todas las probabilidades &lt; 30%
              </div>
            </div>""", unsafe_allow_html=True)
    with col_neg:
        st.markdown(
            "<div style='color:#4d6a99;font-size:0.82rem;margin-bottom:0.8rem;'>"
            "Patologías no detectadas</div>",
            unsafe_allow_html=True)
        st.markdown("".join(prob_bar(p,v,show_info=False) for p,v in negativos),
                    unsafe_allow_html=True)

    # ── Info del modelo (compacta)
    st.markdown("---")
    section_title("🧠", "Información del modelo")
    _model_cards(ckpt, test_results)

    # ── Descarga PDF
    st.markdown("---")
    section_title("📄", "Descargar reporte",
                  "PDF con imagen original, Grad-CAM y tabla completa de resultados.")
    with st.spinner("Preparando PDF…"):
        pdf_bytes = build_pdf(
            img_pil, pil_overlay, probs,
            uploaded.name, ckpt or {}, test_results)
    st.download_button(
        "⬇  Descargar reporte PDF",
        data=pdf_bytes,
        file_name=f"reporte_rx_{uploaded.name.rsplit('.',1)[0]}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════
#  SECCIÓN: EDA
# ══════════════════════════════════════════════════════════
def section_eda():
    section_title(
        "📈", "Análisis Exploratorio del Dataset",
        "NIH ChestX-ray14 — 112,120 radiografías · 14 patologías · 30,805 pacientes",
    )

    # Estadísticas clave en cards
    cols = st.columns(4, gap="small")
    stats = [
        ("Imágenes totales", "112,120", "radiografías frontales"),
        ("Pacientes únicos", "30,805", "sin solapamiento train/test"),
        ("Patologías",       "14",     "clasificación multi-etiqueta"),
        ("Sin hallazgo",     "53.8 %", "60,361 imágenes normales"),
    ]
    for col, (lbl, val, sub) in zip(cols, stats):
        with col:
            st.markdown(card(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Tabs para agrupar las figuras
    tab1, tab2, tab3 = st.tabs([
        "📊 Distribución de patologías",
        "👥 Demografía y pacientes",
        "🔗 Co-ocurrencia y edad",
    ])

    with tab1:
        show_figure(EDA_FIGURES[0])  # distribución patologías
        st.markdown("---")
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            show_figure(EDA_FIGURES[1])  # no finding vs finding
        with col_b:
            show_figure(EDA_FIGURES[2])  # hallazgos por imagen

    with tab2:
        show_figure(EDA_FIGURES[4])  # demografía
        st.markdown("---")
        show_figure(EDA_FIGURES[6])  # imgs por paciente

    with tab3:
        show_figure(EDA_FIGURES[3])  # co-ocurrencia
        st.markdown("---")
        show_figure(EDA_FIGURES[5])  # prevalencia por edad


# ══════════════════════════════════════════════════════════
#  SECCIÓN: EVALUACIÓN DE RESULTADOS
# ══════════════════════════════════════════════════════════
def section_evaluacion(test_results):
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

    # Cards de métricas clave
    auc_mean  = test_results["test_auc_mean"]
    best_ep   = test_results["best_epoch"]
    val_auc   = test_results["best_val_auc"]
    chex_mean = sum(CHEXNET_AUC.values()) / len(CHEXNET_AUC)

    cols = st.columns(4, gap="small")
    with cols[0]: st.metric("Test AUC promedio",     f"{auc_mean:.4f}")
    with cols[1]: st.metric("CheXNet AUC (ref.)",    f"{chex_mean:.4f}")
    with cols[2]: st.metric("Mejor época",            f"{best_ep} / 10")
    with cols[3]: st.metric("Val AUC (mejor época)",  f"{val_auc:.4f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Tabs de evaluación
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Curvas de aprendizaje",
        "🏆 AUC por patología",
        "📈 Curvas ROC",
        "🔍 Distribución de predicciones",
    ])

    with tab1:
        show_figure(EVAL_FIGURES[0])  # curvas aprendizaje

    with tab2:
        # Tabla interactiva
        import pandas as pd
        rows = []
        for p in PATHOLOGIES:
            our  = test_results["test_auc_per_class"][p]
            ref  = CHEXNET_AUC[p]
            rows.append({
                "Patología":     p,
                "Nuestro AUC":   round(our, 4),
                "CheXNet AUC":   round(ref, 4),
                "Δ Diferencia":  round(our - ref, 4),
                "F1-Score":      round(test_results["test_f1_per_class"][p], 4),
                "Avg Precision": round(test_results["test_ap_per_class"][p], 4),
            })
        df = pd.DataFrame(rows)

        def hl(v):
            return ("background-color:#052e16;color:#4ade80;font-weight:600"
                    if v >= 0 else
                    "background-color:#450a0a;color:#f87171;font-weight:600")

        styled = (
            df.style
            .map(hl, subset=["Δ Diferencia"])
            .format({
                "Nuestro AUC":   "{:.4f}",
                "CheXNet AUC":   "{:.4f}",
                "Δ Diferencia":  "{:+.4f}",
                "F1-Score":      "{:.4f}",
                "Avg Precision": "{:.4f}",
            })
        )
        st.dataframe(styled, use_container_width=True, height=530)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Gráfica comparativa generada por el notebook
        show_figure(EVAL_FIGURES[1])  # 09_auc_por_clase.png

    with tab3:
        show_figure(EVAL_FIGURES[2])  # curvas ROC

    with tab4:
        show_figure(EVAL_FIGURES[3])  # distribución predicciones


# ══════════════════════════════════════════════════════════
#  SECCIÓN: ACERCA DEL PROYECTO
# ══════════════════════════════════════════════════════════
def section_acerca():
    section_title("ℹ️", "Acerca del proyecto")
    st.markdown("""
    **Proyecto de Aula — Inteligencia Artificial — III Corte**
    Universidad Popular del Cesar · Facultad de Ingeniería de Sistemas · 2026-I

    | | |
    |---|---|
    | **Integrantes** | Mateo Lopez Patiño · Anaclaudia Vega Martinez · Tonny Enrique Jimenez Marquez |
    | **Modelo** | DenseNet-121 con Transfer Learning (ImageNet → NIH ChestX-ray14) |
    | **Dataset** | NIH ChestX-ray14 — 112,120 imágenes · 14 patologías · 30,805 pacientes |
    | **Entrenamiento** | 10 épocas · GPU Tesla T4 (Kaggle) · Adam lr=1e-4 · BCE multi-etiqueta |
    | **Mejor val AUC** | 0.8178 (época 7) · Test AUC medio: 0.7938 |

    ---

    ### Flujo de procesamiento
    1. **Carga** — usuario sube PNG o JPG.
    2. **Preprocesamiento** — escala de grises → RGB → 224×224 → normalización ImageNet.
    3. **Inferencia** — DenseNet-121 produce 14 probabilidades independientes (sigmoid).
    4. **Grad-CAM** — calculado sobre `denseblock4`, el último bloque convolucional.
    5. **Reporte** — PDF con imagen, mapa de calor y tabla completa de resultados.

    ### Umbral de detección
    Se usa **30 %** como umbral para marcar un hallazgo como presente (no 50 %)
    porque el dataset NIH tiene fuerte desbalance entre clases positivas y negativas.

    ### Advertencia clínica
    > Este sistema es una herramienta de **apoyo académico**.
    > No reemplaza el criterio de un médico radiólogo especializado.
    > Los resultados no deben usarse para diagnóstico clínico real.

    ---

    ### Referencias
    - Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection.* arXiv:1711.05225
    - Wang et al. (2017). *ChestX-ray8: Hospital-Scale Chest X-Ray Database.* IEEE CVPR.
    - Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
    - Huang et al. (2017). *Densely Connected Convolutional Networks.* IEEE CVPR.
    """)


# ══════════════════════════════════════════════════════════
#  HELPER: cards de modelo
# ══════════════════════════════════════════════════════════
def _model_cards(ckpt, test_results):
    auc_m   = test_results["test_auc_mean"] if test_results else 0.0
    best_ep = test_results["best_epoch"]    if test_results else (ckpt or {}).get("epoch","?")
    val_auc = test_results["best_val_auc"]  if test_results else (ckpt or {}).get("val_auc",0.0)
    chex_m  = sum(CHEXNET_AUC.values()) / len(CHEXNET_AUC)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Arquitectura",         "DenseNet-121")
    with c2: st.metric("Test AUC promedio",    f"{auc_m:.4f}")
    with c3: st.metric("Mejor época",           f"{best_ep} / 10")
    with c4: st.metric("Val AUC (mejor época)", f"{val_auc:.4f}")

    diff_str = (
        '<span style="color:#4ade80">▲ por encima</span>'
        if auc_m >= chex_m else
        '<span style="color:#f97316">▼ por debajo</span>'
    )
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;margin-top:0.8rem;'>
      {card("DATASET","NIH ChestX-ray14",
            "112,120 radiografías · 30,805 pacientes")}
      {card("ENTRENAMIENTO","10 épocas · Tesla T4",
            "Adam lr=1e-4 · BCE multi-etiqueta")}
      {card("REFERENCIA",f"CheXNet {chex_m:.4f}",
            f"Nuestro: {auc_m:.4f} {diff_str}")}
      {card("TÉCNICA","Transfer Learning",
            "ImageNet preentrenado → dominio médico")}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="RX-IA · Tórax",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Sidebar
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

        st.markdown("""
        <hr style='border-color:#1a2640;margin-top:1.5rem;'>
        <div style='font-size:0.68rem;color:#1a3a6b;text-align:center;
                    margin-top:0.8rem;line-height:1.7;'>
          DenseNet-121 · NIH ChestX-ray14<br>
          Universidad Popular del Cesar<br>
          Inteligencia Artificial · 2026-I
        </div>
        """, unsafe_allow_html=True)

    # Carga de modelo y resultados
    model, ckpt  = load_model()
    test_results = load_test_results()

    # Header
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
            f"<span>Época <span style='color:#60a5fa;'>{ckpt.get('epoch','?')} / 10</span></span>"
            f"<span>Val AUC <span style='color:#60a5fa;'>{ckpt.get('val_auc',0):.4f}</span></span>"
            f"<span>14 patologías · NIH ChestX-ray14</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='border-color:#1a2640;margin:0.8rem 0 1rem;'>",
        unsafe_allow_html=True,
    )

    # Alerta si falta el modelo en la sección de análisis
    if model is None and "Análisis" in page:
        st.error(
            "⚠️  No se encontró `outputs/models/best_model.pt`. "
            "Descárgalo de Kaggle y colócalo en esa ruta."
        )
        return

    # Router
    if "Análisis" in page:
        section_analisis(model, ckpt, test_results)
    elif "EDA" in page:
        section_eda()
    elif "Evaluación" in page:
        section_evaluacion(test_results)
    else:
        section_acerca()


if __name__ == "__main__":
    main()