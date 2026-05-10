# Standard Library Imports
import sys
import re
from io import BytesIO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.lib import colors
from src.utils.logger import logging
from src.utils.exception import CustomException
 
class PDFGenerator:
    """
    Markdown-aware PDF Generator for the Multi-Agent Research Report.

    Supports:
      - # / ## / ### headings (mapped to H1 / H2 / H3 styles)
      - Unordered lists ('-' or '*' bullets)
      - Ordered lists ('1.', '2.' ...)
      - **bold** and *italic*
      - Inline links / URLs
      - Cover page, page numbers, and section spacing

    Designed to comfortably exceed 4 pages for a ~2500-3500 word report.
    """

    MAX_REPORT_CHARS = 60000
    MAX_PARAGRAPH_CHARS = 4000

    # Text helpers
    @staticmethod
    def _escape(text: str) -> str:
        """
        Escape characters that ReportLab's mini-HTML parser dislikes.
        """

        if not text:
            return ""
        
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

    @staticmethod
    def _inline_markdown(text: str) -> str:
        """
        Convert inline markdown (**bold**, *italic*, `code`, links) to the
        small subset of HTML understood by ReportLab Paragraph.
        Input MUST already be HTML-escaped.
        """

        if not text:
            return ""

        # Bold: **text** or __text__
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

        # Italic: *text* or _text_  (avoid clashing with bold)
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", text)

        # Inline code: `code`
        text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)

        # Markdown links [text](url)
        text = re.sub(
            r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
            r'<link href="\2" color="blue">\1</link>',
            text,
        )

        # Bare URLs -> clickable
        text = re.sub(
            r'(?<!href=")(https?://[^\s<]+)',
            r'<link href="\1" color="blue">\1</link>',
            text,
        )

        return text

    @classmethod
    def _format(cls, text: str) -> str:
        """Escape HTML-sensitive chars then apply inline markdown."""
        return cls._inline_markdown(cls._escape(text.strip()))

    # Page decoration: page numbers + footer
    @staticmethod
    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.setStrokeColor(colors.lightgrey)
        canvas.line(40, 40, A4[0] - 40, 40)
        canvas.drawString(40, 28, "Multi-Agent Research Report")
        canvas.drawRightString(A4[0] - 40, 28, f"Page {doc.page}")
        canvas.restoreState()

    # Main entry point
    @staticmethod
    def generate_pdf(report_text: str) -> BytesIO:
        try:
            logging.info("PDF generation started.")

            if not report_text or not report_text.strip():
                raise ValueError("Report text is empty.")

            report_text = report_text[: PDFGenerator.MAX_REPORT_CHARS]

            # Strip accidental code fences from the LLM
            report_text = re.sub(r"^```[a-zA-Z]*\n?", "", report_text.strip())
            report_text = re.sub(r"\n?```$", "", report_text.strip())

            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize = A4,
                title = "Multi-Agent Research Report",
                author = "Multi-Agent Research System",
                rightMargin = 0.7 * inch,
                leftMargin = 0.7 * inch,
                topMargin = 0.8 * inch,
                bottomMargin = 0.8 * inch,
            )

            styles = getSampleStyleSheet()

            cover_title_style = ParagraphStyle(
                name = "CoverTitle",
                parent = styles["Title"],
                fontSize = 28,
                leading = 34,
                alignment = TA_CENTER,
                spaceAfter = 24,
                textColor = colors.HexColor("#0B3D91"),
            )
            cover_sub_style = ParagraphStyle(
                name = "CoverSub",
                parent = styles["Normal"],
                fontSize = 13,
                leading = 18,
                alignment = TA_CENTER,
                textColor = colors.HexColor("#444444"),
                spaceAfter = 10,
            )
            h1_style = ParagraphStyle(
                name = "H1",
                parent = styles["Heading1"],
                fontSize = 20,
                leading = 26,
                spaceBefore = 18,
                spaceAfter = 12,
                textColor = colors.HexColor("#0B3D91"),
            )
            h2_style = ParagraphStyle(
                name = "H2",
                parent = styles["Heading2"],
                fontSize = 15,
                leading = 20,
                spaceBefore = 14,
                spaceAfter = 8,
                textColor = colors.HexColor("#11479E"),
            )
            h3_style = ParagraphStyle(
                name = "H3",
                parent = styles["Heading3"],
                fontSize = 12,
                leading = 16,
                spaceBefore = 10,
                spaceAfter = 6,
                textColor = colors.HexColor("#333333"),
            )
            body_style = ParagraphStyle(
                name = "Body",
                parent = styles["BodyText"],
                fontSize = 10.5,
                leading = 15,
                alignment = TA_JUSTIFY,
                spaceAfter = 8,
            )
            bullet_style = ParagraphStyle(
                name = "Bullet",
                parent = body_style,
                leftIndent = 14,
                bulletIndent = 2,
                spaceAfter = 4,
            )

            story = []

            # ----- Cover page -------------------------------------------
            title_match = re.search(r"^\s*#\s+(.+)$", report_text, re.MULTILINE)
            report_title = (
                title_match.group(1).strip()
                if title_match
                else "Multi-Agent Research Report"
            )

            story.append(Spacer(1, 1.4 * inch))
            story.append(Paragraph(PDFGenerator._format(report_title), cover_title_style))
            story.append(HRFlowable(width="60%", thickness=1.2,
                                    color=colors.HexColor("#0B3D91"),
                                    spaceBefore=6, spaceAfter=18,
                                    hAlign="CENTER"))
            story.append(Paragraph(
                "Generated by the Multi-Agent Research &amp; Report Generator",
                cover_sub_style,
            ))
            story.append(Paragraph(
                "Agents: Orchestrator &middot; Search &middot; Extraction &middot; Writer &middot; Reviewer",
                cover_sub_style,
            ))
            story.append(Spacer(1, 0.6 * inch))
            story.append(Paragraph(
                f"Date: {datetime.now().strftime('%B %d, %Y')}",
                cover_sub_style,
            ))
            story.append(PageBreak())

            # ----- Body parsing -----------------------------------------
            lines = report_text.splitlines()

            # Skip the first H1 we already used on the cover (avoid duplication)
            if title_match:
                for idx, ln in enumerate(lines):
                    s = ln.strip()
                    if s.startswith("# ") and s[2:].strip() == report_title:
                        lines.pop(idx)
                        break

            i = 0
            n = len(lines)
            while i < n:
                stripped = lines[i].strip()

                if not stripped:
                    story.append(Spacer(1, 4))
                    i += 1
                    continue

                # Headings
                if stripped.startswith("### "):
                    story.append(Paragraph(PDFGenerator._format(stripped[4:]), h3_style))
                    i += 1
                    continue
                if stripped.startswith("## "):
                    story.append(Paragraph(PDFGenerator._format(stripped[3:]), h2_style))
                    i += 1
                    continue
                if stripped.startswith("# "):
                    story.append(Paragraph(PDFGenerator._format(stripped[2:]), h1_style))
                    i += 1
                    continue

                # Unordered list
                if re.match(r"^[-*+]\s+", stripped):
                    items = []
                    while i < n and re.match(r"^[-*+]\s+", lines[i].strip()):
                        item_text = re.sub(r"^[-*+]\s+", "", lines[i].strip())
                        items.append(
                            ListItem(
                                Paragraph(PDFGenerator._format(item_text), bullet_style),
                                leftIndent=14,
                            )
                        )
                        i += 1
                    story.append(
                        ListFlowable(items, bulletType="bullet", start="•",
                                     leftIndent=18, bulletFontSize=9)
                    )
                    story.append(Spacer(1, 6))
                    continue

                # Ordered list
                if re.match(r"^\d+[.)]\s+", stripped):
                    items = []
                    while i < n and re.match(r"^\d+[.)]\s+", lines[i].strip()):
                        item_text = re.sub(r"^\d+[.)]\s+", "", lines[i].strip())
                        items.append(
                            ListItem(
                                Paragraph(PDFGenerator._format(item_text), bullet_style),
                                leftIndent=14,
                            )
                        )
                        i += 1
                    story.append(
                        ListFlowable(items, bulletType="1",
                                     leftIndent=18, bulletFontSize=9)
                    )
                    story.append(Spacer(1, 6))
                    continue

                # Horizontal rule
                if re.match(r"^[-*_]{3,}$", stripped):
                    story.append(HRFlowable(width="100%", thickness=0.5,
                                            color=colors.lightgrey,
                                            spaceBefore=6, spaceAfter=6))
                    i += 1
                    continue

                # Paragraph: gather consecutive non-empty, non-special lines
                para_lines = [stripped]
                i += 1
                while i < n:
                    nxt = lines[i].strip()
                    if (
                        not nxt
                        or nxt.startswith("#")
                        or re.match(r"^[-*+]\s+", nxt)
                        or re.match(r"^\d+[.)]\s+", nxt)
                        or re.match(r"^[-*_]{3,}$", nxt)
                    ):
                        break
                    para_lines.append(nxt)
                    i += 1
                paragraph = " ".join(para_lines)

                for cs in range(0, len(paragraph), PDFGenerator.MAX_PARAGRAPH_CHARS):
                    chunk = paragraph[cs: cs + PDFGenerator.MAX_PARAGRAPH_CHARS]
                    story.append(Paragraph(PDFGenerator._format(chunk), body_style))

            if len(story) < 6:
                story.append(Paragraph(
                    "<i>Note: the generated content was unusually short. "
                    "Try re-running the workflow with a more specific topic.</i>",
                    body_style,
                ))

            logging.info("Building PDF document with %d flowables...", len(story))
            doc.build(
                story,
                onFirstPage=PDFGenerator._on_page,
                onLaterPages=PDFGenerator._on_page,
            )

            buffer.seek(0)
            logging.info("PDF generation completed.")
            return buffer

        except Exception as e:
            logging.exception("PDF generation failed.")
            raise CustomException(e, sys)
