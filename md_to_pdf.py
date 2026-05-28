"""
Convierte archivos Markdown a PDF con estilo academico.
Uso: python md_to_pdf.py docs/01_fase_analisis.md
"""

import sys
import re
from pathlib import Path
import markdown
from xhtml2pdf import pisa

CSS = """
@page {
    size: A4;
    margin: 2.5cm 2.2cm 2.5cm 2.2cm;
}

body {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a1a;
}

h1 {
    font-size: 17pt;
    color: #1a3a5c;
    border-bottom: 2.5px solid #1a3a5c;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
}

h2 {
    font-size: 13.5pt;
    color: #1a3a5c;
    border-bottom: 1px solid #b0c4de;
    padding-bottom: 3px;
    margin-top: 22px;
    margin-bottom: 8px;
}

h3 {
    font-size: 11.5pt;
    color: #2e5f8a;
    margin-top: 16px;
    margin-bottom: 6px;
}

h4 {
    font-size: 10.5pt;
    color: #444;
    font-style: italic;
    margin-top: 12px;
    margin-bottom: 4px;
}

p {
    margin: 6px 0 10px 0;
    text-align: justify;
}

blockquote {
    border-left: 4px solid #b0c4de;
    margin: 10px 0 10px 10px;
    padding: 4px 12px;
    color: #444;
    font-style: italic;
    background-color: #f5f8fc;
}

code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 1px 4px;
    border-radius: 2px;
    color: #c0392b;
}

pre {
    background-color: #f4f6f8;
    border: 1px solid #dde3ea;
    border-left: 4px solid #2e5f8a;
    padding: 10px 14px;
    font-size: 8.5pt;
    font-family: "Courier New", monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 10px 0;
    line-height: 1.4;
}

pre code {
    background: none;
    padding: 0;
    color: #1a1a1a;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
}

th {
    background-color: #1a3a5c;
    color: white;
    padding: 7px 10px;
    text-align: left;
    font-weight: bold;
}

td {
    padding: 6px 10px;
    border-bottom: 1px solid #dde3ea;
    vertical-align: top;
}

tr:nth-child(even) td {
    background-color: #f5f8fc;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 18px 0;
}

ul, ol {
    margin: 6px 0 10px 0;
    padding-left: 22px;
}

li {
    margin-bottom: 3px;
}

strong {
    color: #1a1a1a;
}

a {
    color: #2e5f8a;
}
"""

def convert_md_to_pdf(md_path: str, output_path: str = None):
    md_file = Path(md_path)
    if not md_file.exists():
        print(f"ERROR: No se encontro el archivo: {md_path}")
        sys.exit(1)

    if output_path is None:
        output_path = md_file.with_suffix(".pdf")
    else:
        output_path = Path(output_path)

    print(f"Leyendo: {md_file.name}")
    text = md_file.read_text(encoding="utf-8")

    print("Convirtiendo Markdown -> HTML...")
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "nl2br", "attr_list"]
    )
    body_html = md.convert(text)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8"/>
    <style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""

    # Resolver imagenes relativas desde la carpeta del .md
    base_url = md_file.parent.resolve().as_uri() + "/"

    print(f"Generando PDF: {output_path.name} ...")
    with open(output_path, "wb") as out_file:
        status = pisa.CreatePDF(html, dest=out_file, encoding="utf-8", base_url=base_url)

    if status.err:
        print(f"ERROR al generar PDF: {status.err}")
        sys.exit(1)

    size_kb = output_path.stat().st_size / 1024
    print(f"PDF generado exitosamente: {output_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python md_to_pdf.py <archivo.md> [salida.pdf]")
        print("Ejemplo: python md_to_pdf.py docs/01_fase_analisis.md")
        sys.exit(1)

    md_input = sys.argv[1]
    pdf_output = sys.argv[2] if len(sys.argv) > 2 else None
    convert_md_to_pdf(md_input, pdf_output)
