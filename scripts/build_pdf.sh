#!/usr/bin/env bash
# Build report/build/main.pdf with a local LaTeX installation (MiKTeX or TeX Live).
#   bash scripts/build_pdf.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$ROOT/report/build"
mkdir -p "$B/figures" "$B/tables"
cp "$ROOT/report/main.tex" "$ROOT/report/references.bib" "$ROOT"/report/style/* "$B/"
cp "$ROOT"/figures/*.pdf "$B/figures/"
cp "$ROOT"/results/tables/*.tex "$B/tables/"
cd "$B"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
grep -E "Warning.*(undefined|Citation)" main.log | sort -u || true
echo "Built $B/main.pdf"
