# ============================================================================
# HybridRAG -- HTML Content Parser (src/parsers/html_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Extracts readable text from HTML content. Used by:
#     1. HttpParser (fetches + parses web pages from intranet/URLs)
#     2. EmlParser (could use this for HTML email bodies)
#     3. Any future module that receives HTML content
#
# WHY SEPARATE FROM HTTP PARSER:
#   HTML parsing (stripping tags, extracting text) is a pure function
#   with ZERO network access. Keeping it separate from the HTTP fetcher
#   means it can be tested without network mocks and reused anywhere.
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import re
from html.parser import HTMLParser as _StdlibHTMLParser
from typing import List, Tuple, Dict, Any


# Tags whose content should be completely discarded (not just the tags)
_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "svg", "math", "template",
    "head", "iframe", "object", "embed",
})

# Block-level tags that should produce paragraph breaks
_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "main", "aside", "header", "footer",
    "nav", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
    "ul", "ol", "li", "table", "tr", "td", "th", "caption",
    "dl", "dt", "dd", "figure", "figcaption", "details", "summary",
    "form", "fieldset", "hr", "br",
})


class _TextExtractor(_StdlibHTMLParser):
    """
    Stdlib-based HTML text extractor. No external dependencies.

    Walks the HTML tree and collects visible text content, skipping
    script/style blocks and converting block elements to line breaks.
    """

    def __init__(self):
        super().__init__()
        self._pieces: List[str] = []
        self._skip_depth = 0
        self._title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._pieces.append("\n")
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._pieces.append("\n")
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title = data.strip()
        if self._skip_depth > 0:
            return
        self._pieces.append(data)

    def handle_entityref(self, name: str) -> None:
        # &amp; &lt; &gt; &nbsp; etc
        entity_map = {"amp": "&", "lt": "<", "gt": ">", "nbsp": " ",
                       "quot": '"', "apos": "'"}
        self._pieces.append(entity_map.get(name, f"&{name};"))

    def handle_charref(self, name: str) -> None:
        try:
            if name.startswith("x"):
                char = chr(int(name[1:], 16))
            else:
                char = chr(int(name))
            self._pieces.append(char)
        except (ValueError, OverflowError):
            pass

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        # Collapse runs of whitespace/newlines
        text = re.sub(r'\n{3,}', '\n\n', raw)
        text = re.sub(r'[ \t]+', ' ', text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()

    def get_title(self) -> str:
        return self._title


def extract_text_from_html(
    html: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract readable text from an HTML string.

    Args:
        html: Raw HTML content string.

    Returns:
        (text, details) where text is the extracted readable content
        and details contains metadata like title and character counts.
    """
    details: Dict[str, Any] = {
        "parser": "HTMLParser",
        "html_length": len(html),
    }

    try:
        extractor = _TextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        title = extractor.get_title()

        details["title"] = title
        details["text_length"] = len(text)
        details["compression_ratio"] = (
            round(len(text) / len(html), 2) if html else 0
        )

        return text, details

    except Exception as e:
        details["error"] = f"HTML parse error: {type(e).__name__}: {e}"
        # Fallback: strip tags with regex (crude but better than nothing)
        fallback = re.sub(r'<[^>]+>', ' ', html)
        fallback = re.sub(r'\s+', ' ', fallback).strip()
        details["fallback"] = True
        return fallback, details
