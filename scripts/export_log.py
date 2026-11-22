"""Convert docs/PROJECT_LOG.md into docs/PROJECT_LOG.pdf (A4, title page, page numbers).

Tables and the figures referenced in the log (relative paths, as in the Markdown file) are
included. Uses `markdown` and `xhtml2pdf` (pure Python, pip-installable on Windows).

Other Markdown documents (e.g. docs/TECHNICAL_REPORT.md) can be exported with the same layout by
passing the file and, optionally, a title, a subtitle, a table of contents (--toc) and a page
break before every second-level section (--h2-pagebreak). Without arguments the behaviour is
unchanged.

Usage:
    python scripts/export_log.py
    python scripts/export_log.py docs/TECHNICAL_REPORT.md --title "..." --subtitle "..." --toc --h2-pagebreak
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

import markdown
from PIL import Image
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PROJECT_LOG.md"
OUT = ROOT / "docs" / "PROJECT_LOG.pdf"

TITLE = "DiffFLIM – Project Log"
SUBTITLE = ("Conditional diffusion for fluorescence lifetime and intensity estimation from low-photon "
            "TCSPC data, with calibrated per-pixel uncertainty")
AUTHOR = "Matheshwaran Prakash"
AFFILIATION = "Otto-von-Guericke-Universität Magdeburg"

CONTENT_WIDTH_CM = 16.0   # A4 width 21 cm minus 2.5 cm margins on both sides
MAX_FIG_HEIGHT_CM = 11.0

# A Unicode TrueType font is needed for τ, ±, ≈, √, ᾱ etc.; the built-in Helvetica lacks them.
# The fonts are registered with reportlab directly: xhtml2pdf's own @font-face loading fails on
# Windows (it cannot reopen its temporary copy of the font file).
FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
     "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
]


def register_font():
    """Register a Unicode font family as 'logfont'; return the CSS family name to use."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from xhtml2pdf import default

    for files in FONT_CANDIDATES:
        if all(Path(f).exists() for f in files):
            names = ["LogFont", "LogFont-Bold", "LogFont-Italic", "LogFont-BoldItalic"]
            for name, f in zip(names, files):
                pdfmetrics.registerFont(TTFont(name, f))
            pdfmetrics.registerFontFamily("LogFont", normal=names[0], bold=names[1],
                                          italic=names[2], boldItalic=names[3])
            default.DEFAULT_FONT["logfont"] = "LogFont"
            return "logfont"
    print("Warning: no Unicode font found, falling back to Helvetica (Greek letters may be missing).")
    return "Helvetica"


def size_images(html):
    """Resolve relative image paths (relative to docs/) and give each image an explicit size
    that fits the page while keeping its aspect ratio (xhtml2pdf does not do this itself)."""
    def repl(m):
        attrs = m.group(1)
        src = re.search(r'src="([^"]+)"', attrs).group(1)
        path = (SRC.parent / src).resolve()
        if not path.exists():
            print(f"Warning: figure not found: {src}")
            return f"<p><i>[missing figure: {src}]</i></p>"
        w, h = Image.open(path).size
        width = CONTENT_WIDTH_CM
        height = width * h / w
        if height > MAX_FIG_HEIGHT_CM:
            width, height = width * MAX_FIG_HEIGHT_CM / height, MAX_FIG_HEIGHT_CM
        alt = re.search(r'alt="([^"]*)"', attrs)
        caption = f'<div class="caption">{alt.group(1)}</div>' if alt and alt.group(1) else ""
        return (f'<div class="figure"><img src="{path.as_posix()}" '
                f'style="width:{width:.2f}cm; height:{height:.2f}cm;" />{caption}</div>')
    return re.sub(r"<img([^>]*)/?>", repl, html)


def build_html(md_text, title=TITLE, subtitle=None, toc=False, h2_pagebreak=False):
    # Drop the Markdown H1; the title page carries the title.
    md_text = re.sub(r"\A#\s+.*\n", "", md_text.lstrip())
    # Markdown needs a blank line before a list; the log writes lists directly under "Changes:".
    md_text = re.sub(r"(?m)^(?!\s*[-*] )(\S[^\n]*)\n(- )", r"\1\n\n\2", md_text)
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists", "fenced_code"])
    body = size_images(body)
    # Keep numbers and their unit together ("2.6 %", "60 s") so they never break across lines.
    body = re.sub(r"(\d) (%|ns|s\b|MB|GB|cm)", r"\1&nbsp;\2", body)
    # Images are wrapped in <p> by markdown; keep figures as blocks.
    body = re.sub(r"<p>\s*(<div class=\"figure\">.*?</div>(?:<div class=\"caption\">.*?</div>)?</div>)\s*</p>",
                  r"\1", body, flags=re.S)
    # Every "Log" entry (h3) except the first starts on a new page for readability.
    if h2_pagebreak:
        body = re.sub(r"<h2>", "<pdf:nextpage /><h2>", body)
    else:
        body = body.replace("<h2>Decisions</h2>", "<pdf:nextpage /><h2>Decisions</h2>")
        body = body.replace("<h2>Log</h2>", "<pdf:nextpage /><h2>Log</h2>")
    # Bold the template field names at the start of a paragraph ("Goal:", "Results:", ...).
    body = re.sub(r"<p>(Goal|Changes|Settings|Results|Compared to before|Problems and fixes|Next):",
                  r'<p><span class="field">\1:</span>', body)

    family = register_font()
    date = datetime.date.today().isoformat()
    css = f"""
    @page {{
        size: a4 portrait;
        margin: 2.2cm 2.5cm 2.4cm 2.5cm;
    }}
    @page content {{
        size: a4 portrait;
        margin: 2.2cm 2.5cm 2.4cm 2.5cm;
        @frame footer {{
            -pdf-frame-content: footer;
            bottom: 1.0cm; margin-left: 2.5cm; margin-right: 2.5cm; height: 0.8cm;
        }}
    }}
    body {{ font-family: {family}; font-size: 10pt; line-height: 1.35; color: #222222; }}
    h2 {{ font-size: 15pt; color: #1f3a5f; border-bottom: 1px solid #1f3a5f;
          padding-bottom: 2pt; margin-top: 14pt; margin-bottom: 6pt; }}
    h3 {{ font-size: 12pt; color: #1f3a5f; margin-top: 14pt; margin-bottom: 4pt; }}
    p {{ margin: 0 0 6pt 0; text-align: left; }}
    ul {{ margin-top: 0; margin-bottom: 6pt; }}
    li {{ margin-bottom: 2pt; }}
    code {{ font-family: Courier; font-size: 9pt; }}
    pre {{ font-family: Courier; font-size: 7.5pt; line-height: 1.2; background-color: #f3f5f8;
           border: 0.5pt solid #c8d0da; padding: 4pt 5pt 4pt 5pt; margin: 2pt 0 8pt 0; }}
    pre code {{ font-size: 7.5pt; }}
    h4 {{ font-size: 10.5pt; color: #1f3a5f; margin-top: 10pt; margin-bottom: 3pt; }}
    pdftoc {{ color: #222222; }}
    pdftoc.pdftoclevel0 {{ font-weight: bold; margin-top: 4pt; font-size: 10pt; }}
    pdftoc.pdftoclevel1 {{ margin-left: 1.0em; font-size: 9pt; }}
    table {{ border: 0.5pt solid #999999; margin: 4pt 0 8pt 0; }}
    th {{ background-color: #e8edf3; font-weight: bold; font-size: 8.5pt;
          padding: 3pt 4pt 2pt 4pt; border: 0.5pt solid #999999; text-align: center; }}
    td {{ font-size: 8.5pt; padding: 3pt 4pt 2pt 4pt; border: 0.5pt solid #999999; text-align: center; }}
    .figure {{ text-align: center; margin: 6pt 0 10pt 0; }}
    .caption {{ font-size: 8.5pt; color: #555555; font-style: italic; margin-top: 2pt; }}
    .field {{ font-weight: bold; color: #1f3a5f; }}
    #footer {{ text-align: center; font-size: 8.5pt; color: #666666; }}
    .title {{ text-align: center; }}
    """
    title_page = f"""
    <div class="title">
      <p style="font-size:1pt;">&nbsp;</p>
      <p style="margin-top:6cm; font-size:{26 if len(title) < 40 else 20}pt; font-weight:bold; color:#1f3a5f;">{title}</p>
      <p style="margin-top:0.8cm; font-size:12pt;">{subtitle or SUBTITLE}</p>
      <p style="margin-top:2.5cm; font-size:14pt;">{AUTHOR}</p>
      <p style="font-size:12pt;">{AFFILIATION}</p>
      <p style="margin-top:1.5cm; font-size:10pt; color:#666666;">Exported {date} from {SRC.relative_to(ROOT).as_posix()}</p>
    </div>
    <pdf:nexttemplate name="content" />
    <pdf:nextpage />
    """
    if toc:
        title_page += ('<p style="font-size:15pt; font-weight:bold; color:#1f3a5f; margin-bottom:8pt;">Contents</p>'
                       '<pdf:toc />')
    short = title.split(":")[0]
    footer = f'<div id="footer">{short} &nbsp;·&nbsp; page <pdf:pagenumber></pdf:pagenumber> of <pdf:pagecount></pdf:pagecount></div>'
    return f"""<html><head><meta charset="utf-8"><style>{css}</style></head>
    <body>{footer}{title_page}{body}</body></html>"""


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=str(SRC), help="Markdown file (default: docs/PROJECT_LOG.md)")
    ap.add_argument("--out", default=None, help="PDF file (default: same name as the source, .pdf)")
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--toc", action="store_true", help="table of contents after the title page")
    ap.add_argument("--h2-pagebreak", action="store_true", help="start every '##' section on a new page")
    args = ap.parse_args()
    SRC = Path(args.source).resolve()
    out = Path(args.out).resolve() if args.out else SRC.with_suffix(".pdf")
    html = build_html(SRC.read_text(encoding="utf-8"), args.title, args.subtitle, args.toc, args.h2_pagebreak)
    with open(out, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if result.err:
        sys.exit(f"PDF export failed with {result.err} error(s).")
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
