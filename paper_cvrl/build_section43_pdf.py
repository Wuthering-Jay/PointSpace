"""Build the illustrated Section 4.3 report PDF from its Markdown source.

Run directly in the IDE. The Markdown report and PNG figures remain the source
of truth; standalone figure PDFs are intentionally not generated.
"""

from __future__ import annotations

from datetime import date
from html import escape
from io import BytesIO
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageTemplate,
    Paragraph,
    Spacer,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "CVRL_4.3_实验分析报告.md"
OUTPUT_PATH = BASE_DIR / "CVRL_4.3_实验分析报告.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simsun.ttc")
BOLD_FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT = RIGHT = 43
TOP = 47
BOTTOM = 49
CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT
NAVY = colors.HexColor("#183452")
BLUE = colors.HexColor("#336A9A")
TEXT = colors.HexColor("#1D2936")
MUTED = colors.HexColor("#617182")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("SimSun", str(FONT_PATH), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("SimSunBold", str(BOLD_FONT_PATH)))
    pdfmetrics.registerFontFamily(
        "SimSun", normal="SimSun", bold="SimSunBold",
        italic="SimSun", boldItalic="SimSunBold",
    )


def make_styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "Body", fontName="SimSun", fontSize=9.05, leading=14.1,
        textColor=TEXT, spaceAfter=6.4, wordWrap="CJK",
        allowWidows=0, allowOrphans=0,
    )
    return {
        "body": base,
        "title": ParagraphStyle(
            "Title", parent=base, fontName="SimSunBold", fontSize=19.3,
            leading=27, textColor=NAVY, spaceAfter=9, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Section", parent=base, fontName="SimSunBold", fontSize=13.4,
            leading=19, textColor=NAVY, spaceBefore=17, spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Subsection", parent=base, fontName="SimSunBold", fontSize=10.7,
            leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=5,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base, leftIndent=15, firstLineIndent=-11,
            spaceAfter=3.7,
        ),
        "quote": ParagraphStyle(
            "Quote", parent=base, leftIndent=16, rightIndent=13,
            textColor=colors.HexColor("#3E556C"),
            borderColor=colors.HexColor("#AFC1D2"), borderWidth=1.4,
            borderPadding=(4, 0, 4, 9), spaceBefore=6, spaceAfter=10,
        ),
        "table_header": ParagraphStyle(
            "TableHead", parent=base, fontName="SimSunBold", fontSize=7.35,
            leading=10.3, textColor=colors.white, spaceAfter=0,
        ),
        "table": ParagraphStyle(
            "TableCell", parent=base, fontSize=7.15, leading=10.2,
            spaceAfter=0,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base, fontSize=7.6, leading=10.5,
            textColor=MUTED, alignment=TA_CENTER, spaceBefore=4,
            spaceAfter=10,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base, fontSize=8, leading=12,
            textColor=MUTED, spaceAfter=13,
        ),
    }


def inline_xml(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        text = str(node)
        for dash in ("−", "–", "—", "‑"):
            text = text.replace(dash, "-")
        return escape(text)
    inner = "".join(inline_xml(child) for child in node.children)
    if node.name in {"strong", "b"}:
        return f"<b>{inner}</b>"
    if node.name in {"em", "i"}:
        return f"<i>{inner}</i>"
    if node.name == "code":
        return f'<font color="#22537D">{inner}</font>'
    if node.name == "a":
        href = escape(node.get("href", ""), quote=True)
        return f'<link href="{href}" color="#1E6194">{inner}</link>'
    if node.name == "br":
        return "<br/>"
    return inner


def para(node: Tag, style: ParagraphStyle) -> Paragraph:
    content = "".join(inline_xml(child) for child in node.children)
    return Paragraph(content or " ", style)


def image_flowables(node: Tag, styles: dict) -> list[Flowable]:
    src = node.get("src", "")
    path = (BASE_DIR / src).resolve()
    if not path.is_file() or not path.is_relative_to(BASE_DIR):
        raise FileNotFoundError(f"Missing or out-of-scope report figure: {src}")
    with PILImage.open(path) as source:
        source.load()
        original_width, original_height = source.size
        scale = min(1.0, 2000 / original_width)
        if scale < 1:
            source = source.resize(
                (round(original_width * scale), round(original_height * scale)),
                PILImage.Resampling.LANCZOS,
            )
        if source.mode not in {"RGB", "L"}:
            background = PILImage.new("RGB", source.size, "white")
            if "A" in source.getbands():
                background.paste(source, mask=source.getchannel("A"))
            else:
                background.paste(source)
            source = background
        data = BytesIO()
        source.save(data, format="PNG", optimize=True)
        data.seek(0)
    max_height = 342
    ratio = min(CONTENT_WIDTH / original_width, max_height / original_height)
    graphic = Image(data, width=original_width * ratio, height=original_height * ratio)
    graphic.hAlign = "CENTER"
    graphic._source_buffer = data  # Keep bytes alive until document build.
    caption = Paragraph(escape(node.get("alt", "Figure")), styles["caption"])
    return [Spacer(1, 5), graphic, caption]


def table_flowable(node: Tag, styles: dict) -> LongTable:
    rows: list[list[Paragraph]] = []
    for row in node.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        rows.append([
            para(cell, styles["table_header"] if cell.name == "th" else styles["table"])
            for cell in cells
        ])
    if not rows:
        raise ValueError("Empty Markdown table")
    ncol = max(len(row) for row in rows)
    for row in rows:
        row.extend([Paragraph(" ", styles["table"])] * (ncol - len(row)))
    first_share = 0.23 if ncol >= 5 else 0.39
    if ncol == 1:
        widths = [CONTENT_WIDTH]
    else:
        widths = [CONTENT_WIDTH * first_share] + [
            CONTENT_WIDTH * (1 - first_share) / (ncol - 1)
        ] * (ncol - 1)
    table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FA")]),
        ("LINEBELOW", (0, -1), (-1, -1), 0.45, colors.HexColor("#CAD5DF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def page_decoration(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B8C9D8"))
    canvas.setLineWidth(0.55)
    canvas.line(LEFT, PAGE_HEIGHT - 33, PAGE_WIDTH - RIGHT, PAGE_HEIGHT - 33)
    canvas.setFont("SimSun", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, PAGE_HEIGHT - 25, "CVRL | Section 4.3 Swiss-Audit")
    canvas.line(LEFT, 36, PAGE_WIDTH - RIGHT, 36)
    canvas.drawString(LEFT, 24, "Foundation semantics, visibility and return complementarity")
    canvas.drawRightString(PAGE_WIDTH - RIGHT, 24, str(doc.page))
    canvas.restoreState()


def build_story() -> list[Flowable]:
    raw = SOURCE_PATH.read_text(encoding="utf-8")
    html = markdown.markdown(raw, extensions=["tables", "sane_lists", "fenced_code"])
    soup = BeautifulSoup(html, "html.parser")
    styles = make_styles()
    story: list[Flowable] = []
    for node in soup.contents:
        if not isinstance(node, Tag):
            continue
        if node.name == "h1":
            story.append(para(node, styles["title"]))
            story.append(Paragraph(
                f"Swiss-Audit | Illustrated analysis report | {date.today().isoformat()}",
                styles["meta"],
            ))
        elif node.name in {"h2", "h3"}:
            story.append(para(node, styles[node.name]))
        elif node.name == "p":
            image = node.find("img")
            if image is not None:
                story.extend(image_flowables(image, styles))
            else:
                story.append(para(node, styles["body"]))
        elif node.name in {"ul", "ol"}:
            for index, li in enumerate(node.find_all("li", recursive=False), 1):
                prefix = f"{index}." if node.name == "ol" else "•"
                content = "".join(inline_xml(child) for child in li.children)
                story.append(Paragraph(f"{prefix}  {content}", styles["bullet"]))
        elif node.name == "blockquote":
            for paragraph in node.find_all("p", recursive=False):
                story.append(para(paragraph, styles["quote"]))
        elif node.name == "table":
            story.extend([Spacer(1, 4), table_flowable(node, styles), Spacer(1, 8)])
        elif node.name == "hr":
            story.append(Spacer(1, 6))
        else:
            raise ValueError(f"Unhandled Markdown element: {node.name}")
    return story


def build_pdf() -> Path:
    register_fonts()
    document = BaseDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=TOP, bottomMargin=BOTTOM,
        title="CVRL Section 4.3 - Swiss-Audit Analysis",
        author="PointSpace CVRL research",
    )
    document.addPageTemplates(PageTemplate(
        id="Report", frames=[Frame(
            LEFT, BOTTOM, CONTENT_WIDTH, PAGE_HEIGHT - TOP - BOTTOM,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )], onPage=page_decoration,
    ))
    document.build(build_story())
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_pdf())
