"""Constantes compartidas del proyecto (ML + UI)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "data" / "sample_images"

DEFAULT_UMBRAL = 0.30

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

EDA_FIGURES = [
    {
        "file": "01_distribucion_patologias.png",
        "title": "Distribución de las 14 patologías",
        "caption": (
            "Infiltration es la patología más frecuente (17.7 %), seguida de Effusion (11.9 %) "
            "y Atelectasis (10.3 %). Hernia es la más rara con apenas 227 imágenes (0.2 %)."
        ),
    },
    {
        "file": "02_no_finding_vs_finding.png",
        "title": "'No Finding' vs hallazgo positivo",
        "caption": (
            "El 53.8 % de las imágenes no presentan ninguna patología etiquetada. "
            "El 46.2 % restante tiene al menos un hallazgo."
        ),
    },
    {
        "file": "03_hallazgos_por_imagen.png",
        "title": "Cantidad de hallazgos por radiografía",
        "caption": (
            "La mayoría de imágenes positivas tiene 1 o 2 hallazgos simultáneos. "
            "Algunas llegan a tener hasta 9 patologías en una sola radiografía."
        ),
    },
    {
        "file": "04_cooccurrence.png",
        "title": "Co-ocurrencia condicional de patologías",
        "caption": (
            "Edema e Infiltration co-ocurren con frecuencia (43 %). "
            "Cardiomegaly aparece junto a Effusion en el 38 % de los casos."
        ),
    },
    {
        "file": "05_demografia.png",
        "title": "Demografía del dataset",
        "caption": (
            "Mediana de edad: 49 años. 56.5 % masculino. "
            "60 % vista PA, 40 % AP."
        ),
    },
    {
        "file": "06_prevalencia_por_edad.png",
        "title": "Prevalencia por grupo de edad",
        "caption": (
            "Atelectasis y Effusion aumentan con la edad. "
            "Hernia es prácticamente inexistente en menores de 45."
        ),
    },
    {
        "file": "07_imgs_por_paciente.png",
        "title": "Distribución de imágenes por paciente",
        "caption": (
            "La división train/val/test se hizo por paciente para evitar data leakage."
        ),
    },
]

EVAL_FIGURES = [
    {
        "file": "08_curvas_aprendizaje.png",
        "title": "Curvas de aprendizaje — Pérdida y AUC por época",
        "caption": "El mejor AUC de validación (0.8178) se alcanzó en la época 7.",
    },
    {
        "file": "09_auc_por_clase.png",
        "title": "AUC-ROC por patología — Nuestro modelo vs CheXNet",
        "caption": "Hernia, Emphysema y Cardiomegaly son las clases con mejor AUC.",
    },
    {
        "file": "10_curvas_roc.png",
        "title": "Curvas ROC — una por cada patología",
        "caption": "Curvas más alejadas de la diagonal indican mejor clasificación.",
    },
    {
        "file": "11_distribucion_predicciones.png",
        "title": "Distribución de probabilidades — Positivos vs Negativos",
        "caption": "Separación entre distribuciones roja y azul indica poder discriminativo.",
    },
]


def list_sample_images() -> list[Path]:
    """Radiografías de ejemplo para la demo (sin subir archivo)."""
    paths: list[Path] = []
    for pattern in ("*.png", "Implementacion/*.png"):
        paths.extend(sorted(SAMPLE_IMAGES_DIR.glob(pattern)))
    return paths
