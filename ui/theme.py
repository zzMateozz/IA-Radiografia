"""Estilos globales de la app Streamlit."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #060d1a;
    color: #e2e8f0;
}
.stApp { background: #060d1a; }

.disclaimer-banner {
    background: linear-gradient(90deg, #422006 0%, #713f12 100%);
    border: 1px solid #b45309;
    border-radius: 10px;
    padding: 0.65rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #fde68a;
    text-align: center;
}

.hero-summary {
    background: linear-gradient(135deg, #0d1f3c 0%, #1a2640 100%);
    border: 1px solid #1a3a6b;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

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
[data-testid="stStatusWidget"] {
    background: #0d1f3c !important;
    border: 1px solid #1a3a6b !important;
    border-radius: 10px !important;
}
</style>
"""


def disclaimer_banner() -> str:
    return """
    <div class="disclaimer-banner">
      ⚠️ <b>Uso académico exclusivo.</b> Este sistema NO sustituye el criterio de un
      médico radiólogo. No utilizar para diagnóstico clínico real.
    </div>
    """
