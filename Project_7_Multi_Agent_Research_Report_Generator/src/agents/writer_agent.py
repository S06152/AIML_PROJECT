import sys
import warnings
from typing import Any, Dict

from langchain_groq import ChatGroq

from src.agents.base_agent import BaseAgent
from src.models.state import ResearchState
from src.utils.logger import logging
from src.utils.exception import CustomException

warnings.filterwarnings("ignore")


_WRITER_PROMPT = """
You are a Senior Research Report Writer.

Goal:
Produce a DETAILED, professional, multi-section research report in Markdown
that will fill at least 4 printed pages (target ~2500-3500 words).

You will be given:
- The research topic
- The research plan
- A list of REPORT DIMENSIONS (topic-specific themes the report MUST cover)
- Extracted key points
- Statistics
- Source URLs

Strict structure (use these EXACT markdown headings, in this order):

# {{Report Title - a concise, descriptive title derived from the topic}}

## 1. Executive Summary
A concise 150-250 word overview of the topic, why it matters, and the
most important takeaways from the report.

## 2. Introduction
Define the topic, scope, and context. Explain why it is relevant in 2025-2026.
At least 2 well-developed paragraphs.

## 3. Background and Context
Historical evolution, major milestones, and the broader landscape.
At least 2 paragraphs.

## 4. Key Findings
Present 6-10 detailed findings as a bulleted list. Each bullet must be a
full, informative sentence (NOT one or two words). Use the provided
key points and elaborate on them.

## 5. Statistical Insights
Present the provided statistics as a bulleted list. For each statistic add
a one-sentence interpretation explaining its significance. If no numerical
statistics are provided, present qualitative metrics or comparisons instead.

## 6. Detailed Analysis
A multi-paragraph analytical discussion organised by the REPORT DIMENSIONS
provided in the input. Create one `### <Dimension>` sub-heading for EACH
dimension and write 1-2 substantive paragraphs under each. Do NOT skip
any dimension. This section must be the longest part of the report.

## 7. Challenges and Risks
Bulleted list of 5-8 challenges, risks, or open issues, each explained
in 1-2 sentences.

## 8. Future Outlook
Forward-looking discussion (2-3 paragraphs) of trends, opportunities,
and predictions for the next 1-5 years.

## 9. Conclusion
A strong closing summary (1-2 paragraphs) that ties everything together.

## 10. References
A numbered list of the source URLs provided. Format each as:
1. https://example.com
If no URLs are provided, write "No external sources were retrieved."

Hard rules:
- Output ONLY the Markdown report (no preamble, no code fences).
- Use ONLY the information provided; do not fabricate URLs or numbers.
- The report must be GENERIC enough to work for ANY topic
  (technology, finance, science, society, sports, history, etc.) - do not
  assume any specific industry; rely on the dimensions provided.
- Be substantive and specific; avoid filler and repetition.
- Aim for ~2500-3500 words total.
"""


class WriterAgent(BaseAgent):

    def __init__(self, llm: ChatGroq) -> None:

        try:
            logging.info("Initializing WriterAgent")

            super().__init__(llm, _WRITER_PROMPT)

            logging.info("WriterAgent initialized.")

        except Exception as e:
            logging.exception("WriterAgent initialization failed.")
            raise CustomException(e, sys)

    def execute(self, state: ResearchState) -> Dict[str, Any]:

        try:
            logging.info("WRITER AGENT START")

            extracted_data = state.get(
                "extracted_data",
                {}
            )

            if not extracted_data:
                raise ValueError(
                    "No extracted data found."
                )

            topic = state.get("topic", "").strip()
            research_plan = state.get("research_plan", "")
            dimensions = state.get("report_dimensions", []) or []

            # Keep more data this time
            key_points = extracted_data.get(
                "key_points",
                []
            )[:10]

            statistics = extracted_data.get(
                "statistics",
                []
            )[:10]

            sources = extracted_data.get(
                "source_urls",
                []
            )[:10]

            # Format inputs as readable bullet lists
            kp_text = "\n".join(f"- {p}" for p in key_points) or "- (none)"
            stat_text = "\n".join(f"- {s}" for s in statistics) or "- (none)"
            src_text = "\n".join(f"- {u}" for u in sources) or "- (none)"
            dim_text = "\n".join(f"- {d}" for d in dimensions) or (
                "- Overview\n- Current State\n- Stakeholders\n"
                "- Statistics\n- Challenges\n- Future Outlook"
            )

            input_text = f"""
Topic:
{topic}

Research Plan:
{research_plan}

Report Dimensions (use ONE ### sub-heading per dimension in Section 6):
{dim_text}

Key Points:
{kp_text}

Statistics:
{stat_text}

Sources:
{src_text}

Now write the full multi-section report following the required structure.
"""

            # Allow a richer prompt
            input_text = input_text[:10000]

            # Generate report
            report = self.run(input_text)

            if not report:
                report = "Report generation failed."

            # Final hard limit (large enough for 4+ pages)
            report = report[:30000]

            logging.info(
                "Report generated successfully."
            )

            logging.info("WRITER AGENT END")

            return {
                "draft_report": report
            }

        except Exception as e:
            logging.exception(
                "Error during WriterAgent execution."
            )

            raise CustomException(e, sys)