import operator
import warnings
from typing import Annotated, List, Dict, Optional
from typing_extensions import TypedDict
warnings.filterwarnings("ignore")

class SearchResult(TypedDict):
    title: str
    snippet: str
    url: str

class ExtractedData(TypedDict):
    key_points: List[str]
    statistics: List[str]
    source_urls: List[str]

class ReviewFeedback(TypedDict):
    approved: bool
    overall_score: int
    suggestions: List[str]

class ResearchState(TypedDict):
    # User Input
    topic: str

    # Planning
    research_plan: str
    search_queries: List[str]

    # Topic-adaptive dimensions the report MUST cover 
    report_dimensions: List[str]

    # Search
    compressed_research: str
    raw_search_results: Annotated[List[SearchResult], operator.add]

    # Extraction
    extracted_data: Optional[ExtractedData]

    # Writing
    draft_report: str
    final_report: str

    # Review
    review_feedback: Optional[ReviewFeedback]
    revision_count: int

    # Errors / Logs
    errors: Annotated[List[str],operator.add]
    messages: Annotated[ List[Dict], operator.add]