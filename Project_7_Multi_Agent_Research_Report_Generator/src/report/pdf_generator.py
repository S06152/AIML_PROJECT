# Standard Library Imports
import sys
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from src.utils.logger import logging
from src.utils.exception import CustomException

class PDFGenerator:
    """
    Utility class to generate PDF from a text-based report.

    Responsibilities:
        - Convert markdown/text report into PDF
        - Apply basic formatting (headings, spacing)
        - Return PDF as in-memory buffer (BytesIO)
    """

    @staticmethod
    def generate_pdf(report_text: str) -> BytesIO:
        """
        Convert report text into a PDF buffer.

        Args:
            report_text (str): Final report content

        Returns:
            BytesIO: PDF file buffer

        Raises:
            CustomException: If PDF generation fails
        """
        try:
            logging.info("PDF GENERATION START")

            # Validate Input
            if not report_text or not report_text.strip():
                raise ValueError("Report text is empty. Cannot generate PDF.")

            logging.info("Report length: %d characters", len(report_text))

            # Initialize PDF Document
            buffer = BytesIO()

            doc = SimpleDocTemplate(buffer, title = "Research Report", author = "Multi-Agent System")

            styles = getSampleStyleSheet()

            # Custom heading style
            heading_style = ParagraphStyle(
                name = "Heading",
                parent = styles["Heading2"],
                spaceAfter = 10
            )

            story = []

            # Convert Text → PDF Elements
            for line in report_text.split("\n"):
                clean_line = line.strip()

                if not clean_line:
                    continue

                # Detect headings (basic markdown support)
                if clean_line.startswith("#"):
                    formatted_text = clean_line.lstrip("#").strip()
                    story.append(Paragraph(formatted_text, heading_style))
                else:
                    story.append(Paragraph(clean_line, styles["Normal"]))

                story.append(Spacer(1, 10))

            # Build PDF
            logging.info("Building PDF document...")
            doc.build(story)

            # Reset buffer pointer
            buffer.seek(0)

            logging.info("PDF generated successfully.")
            logging.info("PDF GENERATION END")

            return buffer

        except Exception as e:
            logging.exception("Error during PDF generation.")
            raise CustomException(e, sys)