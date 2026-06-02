"""Sistema de diseño — paleta clínica moderna, tipografía Inter, sin emojis."""

PALETTE = {
    "bg": "#F2F5F8",
    "bg_alt": "#F8F9FA",
    "surface": "#FFFFFF",
    "surface_soft": "#F0F4F8",
    "border": "#DEE2E6",
    "border_light": "#E9ECEF",
    "text": "#212529",
    "text_muted": "#6C757D",
    "text_light": "#ADB5BD",
    "primary": "#0D6EFD",
    "primary_dark": "#0B5ED7",
    "primary_light": "#6EA8FE",
    "accent": "#198754",
    "accent_warm": "#DC3545",
    "danger": "#DC3545",
    "warning": "#FFC107",
    "success": "#198754",
    "info": "#0DCAF0",
    "shadow": "0 0.5rem 1rem rgba(0,0,0,0.08)",
    "shadow_sm": "0 0.125rem 0.25rem rgba(0,0,0,0.04)",
}

# Icono SVG de pulmón simple (reemplaza el emoji)
LOGO_SVG = """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 4v4"/>
    <path d="M6 12c-2 0-3 2-3 4v4h3v-4h4v4h4v-4h4v4h3v-4c0-2-1-4-3-4"/>
    <circle cx="9" cy="15" r="2"/>
    <circle cx="15" cy="15" r="2"/>
</svg>"""

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {PALETTE['bg']};
    color: {PALETTE['text']};
}}

.stApp {{
    background: {PALETTE['bg']};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {PALETTE['surface']} !important;
    border-right: 1px solid {PALETTE['border']};
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.88rem !important;
    color: {PALETTE['text_muted']} !important;
    padding: 0.4rem 1rem !important;
    border-radius: 0.5rem !important;
    transition: all 0.15s;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {PALETTE['bg_alt']};
}}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {{
    background: {PALETTE['primary']} !important;
    color: white !important;
    font-weight: 600 !important;
}}

/* Tipografía */
h1, h2, h3 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: {PALETTE['text']} !important;
}}

/* Métricas */
[data-testid="metric-container"] {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border_light']};
    border-radius: 12px;
    padding: 1rem;
    box-shadow: {PALETTE['shadow_sm']};
}}
[data-testid="stMetricValue"] {{
    font-family: 'Inter', sans-serif !important;
    color: {PALETTE['primary']} !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {PALETTE['text_muted']} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* Botones */
.stButton>button, .stDownloadButton>button {{
    background: {PALETTE['primary']} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    transition: background 0.15s, box-shadow 0.15s;
    box-shadow: {PALETTE['shadow_sm']};
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background: {PALETTE['primary_dark']} !important;
    box-shadow: 0 4px 12px rgba(13,110,253,0.25);
}}

/* Upload */
[data-testid="stFileUploaderDropzone"] {{
    background: {PALETTE['surface']} !important;
    border: 2px dashed {PALETTE['primary_light']} !important;
    border-radius: 16px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {PALETTE['primary']} !important;
    background: {PALETTE['surface_soft']} !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border_light']};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    box-shadow: {PALETTE['shadow_sm']};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {PALETTE['text_muted']} !important;
    border-radius: 8px !important;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.4rem 0.8rem;
}}
.stTabs [aria-selected="true"] {{
    background: {PALETTE['primary']} !important;
    color: white !important;
    font-weight: 600;
}}

/* Expander, status, slider */
[data-testid="stExpander"] {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border_light']} !important;
    border-radius: 12px !important;
    box-shadow: {PALETTE['shadow_sm']};
}}
hr {{ border-color: {PALETTE['border_light']} !important; }}

/* Selectbox / inputs en sidebar */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {{
    color: {PALETTE['text_muted']} !important;
    font-size: 0.82rem !important;
}}

/* Clases custom */
.disclaimer-banner {{
    background: #FFF3CD;
    border-left: 4px solid {PALETTE['warning']};
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #856404;
    box-shadow: {PALETTE['shadow_sm']};
}}

.hero-summary {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border_light']};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin: 1.2rem 0;
    box-shadow: {PALETTE['shadow_sm']};
    position: relative;
    overflow: hidden;
}}
.hero-summary::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, {PALETTE['primary']}, {PALETTE['accent']});
}}

.page-header {{
    margin-bottom: 0.5rem;
}}
.page-header .eyebrow {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['primary']};
    margin-bottom: 0.3rem;
}}
.page-header h1 {{
    font-size: 2rem !important;
    line-height: 1.15 !important;
    margin: 0 !important;
    color: {PALETTE['text']} !important;
}}
.page-header .subtitle {{
    color: {PALETTE['text_muted']};
    font-size: 0.95rem;
    margin-top: 0.5rem;
    max-width: 42rem;
}}
.chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 0.75rem;
}}
.chip {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border_light']};
    border-radius: 8px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    color: {PALETTE['text_muted']};
    box-shadow: {PALETTE['shadow_sm']};
}}
.chip strong {{ color: {PALETTE['primary']}; }}

.sidebar-brand {{
    text-align: center;
    padding: 1.2rem 0 1rem;
}}
.sidebar-brand .logo-mark {{
    width: 48px; height: 48px;
    margin: 0 auto 0.75rem;
    background: linear-gradient(135deg, {PALETTE['primary']}, {PALETTE['accent']});
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    box-shadow: {PALETTE['shadow_sm']};
}}
.sidebar-brand .logo-text {{
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: {PALETTE['text']};
    letter-spacing: -0.02em;
}}
.sidebar-brand .logo-sub {{
    font-size: 0.72rem;
    color: {PALETTE['text_light']};
    margin-top: 0.2rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
</style>
"""


def disclaimer_banner() -> str:
    return """
    <div class="disclaimer-banner">
      <strong>Aviso académico.</strong> Esta herramienta apoya el aprendizaje en IA médica.
      No sustituye el diagnóstico de un radiólogo especializado.
    </div>
    """


def page_header(title: str, subtitle: str = "", chips: list[tuple[str, str]] | None = None) -> str:
    chips_html = ""
    if chips:
        chips_html = '<div class="chip-row">' + "".join(
            f'<span class="chip">{label}: <strong>{val}</strong></span>'
            for label, val in chips
        ) + "</div>"
    return f"""
    <div class="page-header">
      <div class="eyebrow">UPC · Inteligencia Artificial 2026-I</div>
      <h1>{title}</h1>
      {"<p class='subtitle'>" + subtitle + "</p>" if subtitle else ""}
      {chips_html}
    </div>
    """


def sidebar_brand() -> str:
    p = PALETTE
    return f"""
    <div class="sidebar-brand">
      <div class="logo-mark">{LOGO_SVG}</div>
      <div class="logo-text">PulmoScan</div>
      <div class="logo-sub">Análisis de radiografías</div>
    </div>
    <hr style="border-color:{p['border']}; margin: 0.5rem 0 1rem;">
    """