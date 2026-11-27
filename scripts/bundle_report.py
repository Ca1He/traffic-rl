"""Bundle everything Overleaf needs into report_bundle.zip.

    python scripts/bundle_report.py

Contents: main.tex, references.bib, figures/*.pdf, tables/*.tex.
Upload these into an Overleaf project created from the ICML template (the template provides
icml20XX.sty / .bst); do not upload the zip as a standalone project.
"""
import glob
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(ROOT, "report_bundle.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(ROOT, "report", "main.tex"), "main.tex")
    z.write(os.path.join(ROOT, "report", "references.bib"), "references.bib")
    for f in glob.glob(os.path.join(ROOT, "report", "style", "*")):
        z.write(f, os.path.basename(f))
    for f in glob.glob(os.path.join(ROOT, "figures", "*.pdf")):
        z.write(f, f"figures/{os.path.basename(f)}")
    for f in glob.glob(os.path.join(ROOT, "results", "tables", "*.tex")):
        z.write(f, f"tables/{os.path.basename(f)}")
print("wrote", out)
for name in zipfile.ZipFile(out).namelist():
    print("  ", name)
