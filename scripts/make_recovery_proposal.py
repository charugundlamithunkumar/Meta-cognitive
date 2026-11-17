"""Build the instructor-review PDF from its editable Markdown source (CPU only).

Usage: python3 scripts/make_recovery_proposal.py
Uses the typography of make_report.py without importing its executable report code.
Supports only the headings, paragraphs, bullets, links and tables used in this proposal.
"""

from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "after-the-fix-problem-statement.md"
OUTPUT = ROOT / "after-the-fix-problem-statement.pdf"
INK = colors.HexColor("#203141")
ACCENT = colors.HexColor("#1d4e89")
MUTED = colors.HexColor("#52606d")
RULE = colors.HexColor("#cbd2d9")
WIDTH = A4[0] - 36 * mm


def inline(text):
    """Escape text before translating the small, trusted Markdown subset."""
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                  r'<link href="\2" color="#1d4e89">\1</link>', text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def styles():
    font_dir = Path("/usr/share/fonts/dejavu")
    for suffix, filename in [
        ("", "DejaVuSans.ttf"), ("-B", "DejaVuSans-Bold.ttf"),
        ("-I", "DejaVuSans-Oblique.ttf"),
        ("-BI", "DejaVuSans-BoldOblique.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont("DV" + suffix, str(font_dir / filename)))
    for bold, italic, name in [(0, 0, "DV"), (1, 0, "DV-B"),
                               (0, 1, "DV-I"), (1, 1, "DV-BI")]:
        addMapping("DV", bold, italic, name)
    body = ParagraphStyle("body", fontName="DV", fontSize=9.5,
                          leading=13.3, textColor=INK, spaceAfter=7)
    return {
        "body": body,
        "title": ParagraphStyle("title", parent=body, fontName="DV-B",
                                fontSize=20, leading=24, spaceAfter=5),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=11,
                                   leading=15, textColor=MUTED, spaceAfter=12),
        "heading": ParagraphStyle("heading", parent=body, fontName="DV-B",
                                  fontSize=10.6, leading=14, textColor=ACCENT,
                                  spaceBefore=7, spaceAfter=5, keepWithNext=True),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=10,
                                 bulletIndent=0, spaceAfter=5),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.7,
                               leading=11.5, spaceAfter=0),
    }


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10.5 * mm,
                      "Catching Silent Collapse  |  Proposal for instructor review")
    canvas.drawRightString(A4[0] - 18 * mm, 10.5 * mm, str(doc.page))
    canvas.restoreState()


def make_table(lines, style):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    proportions = [0.24, 0.76] if len(rows[0]) == 2 else [0.15, 0.29, 0.27, 0.29]
    data = [[Paragraph(inline(f"**{c}**" if i == 0 else c), style)
             for c in row] for i, row in enumerate(rows)]
    table = Table(data, colWidths=[WIDTH * p for p in proportions], repeatRows=1)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4ebf3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build():
    style = styles()
    flow = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "<!-- PAGEBREAK -->":
            flow.append(PageBreak())
        elif line.startswith("|"):
            block = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            flow.extend([make_table(block, style["cell"]), Spacer(1, 8)])
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            key = {1: "title", 2: "subtitle", 3: "heading"}[level]
            flow.append(Paragraph(inline(line[level:].strip()), style[key]))
        elif line.startswith("- "):
            flow.append(Paragraph(inline(line[2:]), style["bullet"], bulletText="•"))
        else:
            flow.append(Paragraph(inline(line), style["body"]))
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=20 * mm,
        title="After the Fix: Do LLM Agents Recognize Recovery from Silent Collapse?",
        author="Catching Silent Collapse project",
        subject="Research problem statement and request for instructor approval",
    )
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
