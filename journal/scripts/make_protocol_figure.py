#!/usr/bin/env python3
"""Export the canonical editable protocol diagram to PDF and PNG.

The scientific source is ``journal/results/figures/protocol_ladder.svg``.  It
visualizes the leakage-controlled data roles in Sections 3.1--3.2 of
``journal/manuscript/main.tex``; Table ``tab:rungs`` separately carries the
Shift Ladder and claim boundaries.  The SVG retains editable text and is
designed for full-width manuscript inclusion.

No experimental inputs are consumed.  LibreOffice performs the vector PDF
export, Ghostscript normalizes it to PDF 1.5 for pdfLaTeX compatibility, and
Poppler renders the 300 dpi PNG preview.  The PDF is the manuscript and
submission artifact, while the PNG is only a convenience preview.

The SVG master is authored in Figma and exported with live text, so each styled
run carries an absolute x computed against Inter metrics.  A substituted face
therefore overflows the fitted boxes rather than reflowing, and the export runs
against the Inter faces vendored in ``journal/assets/fonts``
(see ``lc.figure_font_env``) instead of whatever happens to be installed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import lib_common as lc


SVG = lc.FIGURES / "protocol_ladder.svg"
PDF = lc.FIGURES / "protocol_ladder.pdf"
PNG = lc.FIGURES / "protocol_ladder.png"


def require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required command not found: {name}")
    return path


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    if not SVG.is_file():
        raise FileNotFoundError(f"canonical SVG not found: {SVG}")

    libreoffice = require_command("libreoffice")
    ghostscript = require_command("gs")
    pdftocairo = require_command("pdftocairo")

    with tempfile.TemporaryDirectory(prefix="protocol-ladder-") as tmp_name:
        tmp = Path(tmp_name)
        profile = tmp / "libreoffice-profile"
        profile.mkdir()
        run([
            libreoffice,
            "--headless",
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--convert-to",
            "pdf:draw_pdf_Export",
            "--outdir",
            str(tmp),
            str(SVG),
        ], env=lc.figure_font_env(tmp / "xdg"))

        libreoffice_pdf = tmp / "protocol_ladder.pdf"
        if not libreoffice_pdf.is_file():
            raise RuntimeError("LibreOffice did not produce protocol_ladder.pdf")

        normalized_pdf = tmp / "protocol_ladder_pdf15.pdf"
        run([
            ghostscript,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            "-dPDFSETTINGS=/prepress",
            "-dEmbedAllFonts=true",
            "-dSubsetFonts=true",
            "-dAutoRotatePages=/None",
            "-dOmitInfoDate=true",
            "-dOmitID=true",
            "-dOmitXMP=true",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            f"-sOutputFile={normalized_pdf}",
            str(libreoffice_pdf),
        ])
        if not normalized_pdf.is_file():
            raise RuntimeError("Ghostscript did not produce a PDF 1.5 artifact")
        shutil.copyfile(normalized_pdf, PDF)

        preview_stem = tmp / "protocol_ladder"
        run([
            pdftocairo,
            "-png",
            "-singlefile",
            "-r",
            "300",
            str(PDF),
            str(preview_stem),
        ])
        generated_png = preview_stem.with_suffix(".png")
        if not generated_png.is_file():
            raise RuntimeError("pdftocairo did not produce protocol_ladder.png")
        shutil.copyfile(generated_png, PNG)

    print(f"wrote {PDF}")
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
