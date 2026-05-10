# Standard Library Imports
import sys
import re
from io import BytesIO

# ReportLab Imports
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER

# Internal Imports
from src.utils.logger import logging
from src.utils.exception import CustomException


class PDFGenerator:
    """
    Optimized PDF Generator
    """

    MAX_REPORT_CHARS = 25000
    MAX_PARAGRAPH_CHARS = 1500

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text for safe PDF rendering.
        """

        if not text:
            return ""

        # Remove unsupported HTML characters
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")

        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def _split_large_paragraph(
        text: str,
        max_chars: int
    ):
        """
        Split large paragraphs for stability.
        """

        return [
            text[i:i + max_chars]
            for i in range(0, len(text), max_chars)
        ]

    @staticmethod
    def generate_pdf(report_text: str) -> BytesIO:
        """
        Generate PDF from report text.
        """

        try:

            logging.info("PDF generation started.")

            if not report_text or not report_text.strip():
                raise ValueError(
                    "Report text is empty."
                )

            # -----------------------------
            # HARD TOKEN / MEMORY GUARD
            # -----------------------------
            report_text = report_text[
                :PDFGenerator.MAX_REPORT_CHARS
            ]

            report_text = PDFGenerator._clean_text(
                report_text
            )

            # -----------------------------
            # PDF Buffer
            # -----------------------------
            buffer = BytesIO()

            # -----------------------------
            # Document Setup
            # -----------------------------
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                title="Research Report",
                author="Multi-Agent Research System",
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=30
            )

            styles = getSampleStyleSheet()

            # -----------------------------
            # Custom Styles
            # -----------------------------
            title_style = ParagraphStyle(
                name="CustomTitle",
                parent=styles["Heading1"],
                alignment=TA_CENTER,
                spaceAfter=20
            )

            heading_style = ParagraphStyle(
                name="CustomHeading",
                parent=styles["Heading2"],
                spaceAfter=12
            )

            body_style = styles["BodyText"]

            story = []

            # -----------------------------
            # Title
            # -----------------------------
            story.append(
                Paragraph(
                    "Multi-Agent Research Report",
                    title_style
                )
            )

            story.append(Spacer(1, 20))

            # -----------------------------
            # Process Lines
            # -----------------------------
            lines = report_text.split("\n")

            for line in lines:

                clean_line = line.strip()

                if not clean_line:
                    continue

                # -------------------------
                # Headings
                # -------------------------
                if clean_line.startswith("#"):

                    heading = (
                        clean_line
                        .lstrip("#")
                        .strip()
                    )

                    story.append(
                        Paragraph(
                            heading,
                            heading_style
                        )
                    )

                else:

                    # Split huge paragraphs
                    chunks = (
                        PDFGenerator
                        ._split_large_paragraph(
                            clean_line,
                            PDFGenerator.MAX_PARAGRAPH_CHARS
                        )
                    )

                    for chunk in chunks:

                        story.append(
                            Paragraph(
                                chunk,
                                body_style
                            )
                        )

                        story.append(
                            Spacer(1, 8)
                        )

            # -----------------------------
            # Build PDF
            # -----------------------------
            logging.info("Building PDF document...")

            doc.build(story)

            buffer.seek(0)

            logging.info("PDF generation completed.")

            return buffer

        except Exception as e:

            logging.exception(
                "PDF generation failed."
            )

            raise CustomException(e, sys)