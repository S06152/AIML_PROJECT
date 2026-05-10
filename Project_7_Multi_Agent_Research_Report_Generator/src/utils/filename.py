"""
Utility for converting an arbitrary user query (or report title) into a
safe, descriptive PDF filename.

Rules:
- ASCII-only, lowercase
- words separated by underscores
- removes English stop-words so the filename stays meaningful
- strips Windows-reserved characters (< > : " / \\ | ? *)
- max length capped (default 60 chars, excluding the .pdf extension)
- always ends with `.pdf`
- never returns an empty name (falls back to "research_report.pdf")
"""

from __future__ import annotations

import re
import unicodedata

# Small English stop-word list (intentionally short, no nltk dependency).
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "and",
    "or", "but", "with", "without", "is", "are", "was", "were", "be",
    "been", "being", "as", "by", "from", "into", "about", "over",
    "under", "this", "that", "these", "those", "it", "its", "i",
    "you", "we", "they", "he", "she", "do", "does", "did", "how",
    "what", "why", "when", "where", "which", "who", "whom",
}

_MAX_LEN = 60
_DEFAULT = "research_report.pdf"


def _ascii_fold(text: str) -> str:
    """Strip diacritics and drop non-ASCII characters (e.g. Devanagari)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii")


def slugify_filename(
    query: str,
    fallback_title: str | None = None,
    max_len: int = _MAX_LEN,
) -> str:
    """
    Convert a user query (or, if empty, a fallback title) into a safe
    PDF filename.

    Examples
    --------
    >>> slugify_filename("Impact of AI on healthcare in 2026")
    'impact_ai_healthcare_2026.pdf'

    >>> slugify_filename("भारत में AI/banking 2026!!!")
    'ai_banking_2026.pdf'

    >>> slugify_filename("")
    'research_report.pdf'
    """
    raw = (query or "").strip() or (fallback_title or "").strip()
    if not raw:
        return _DEFAULT

    # 1. ASCII-fold (handles accents, devanagari, emojis, etc.)
    text = _ascii_fold(raw).lower()

    # 2. Replace anything that is NOT [a-z0-9] with a space
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # 3. Tokenize, drop stop-words, drop 1-char tokens
    tokens = [t for t in text.split() if t and t not in _STOPWORDS and len(t) > 1]

    # If filtering removed everything, fall back to the un-filtered split
    if not tokens:
        tokens = [t for t in text.split() if t]

    if not tokens:
        return _DEFAULT

    # 4. Greedily build the slug under the length cap
    slug_parts: list[str] = []
    current_len = 0
    for tok in tokens:
        added = len(tok) + (1 if slug_parts else 0)  # +1 for the underscore
        if current_len + added > max_len:
            break
        slug_parts.append(tok)
        current_len += added

    if not slug_parts:           # single token longer than max_len
        slug_parts = [tokens[0][:max_len]]

    return "_".join(slug_parts) + ".pdf"
