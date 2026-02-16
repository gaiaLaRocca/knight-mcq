"""
External knowledge lookup interface for KNIGHT.

External knowledge is replaceable: the default source is Wikipedia.
Implement the ExternalKnowledgeLookup protocol and pass an instance
into term description generation to use a custom source (e.g. URL, file, PDF).
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@runtime_checkable
class ExternalKnowledgeLookup(Protocol):
    """Protocol for looking up external knowledge for a term.

    Implement this interface to plug in a custom source (e.g. URL, file).
    Returns (summary_or_none, is_ambiguous) to match Wikipedia lookup behavior.
    """

    def lookup(
        self,
        term: str,
        context_hint: str | None = None,
        *,
        llm: Any = None,
    ) -> tuple[str | None, bool]:
        """Look up external knowledge for a term.

        Args:
            term: The term to look up.
            context_hint: Optional context (e.g. parent term) for disambiguation.
            llm: Optional LLM instance; some implementations use it for relevance checks.

        Returns:
            Tuple of (summary text or None, is_ambiguous).
        """
        ...


class WikipediaLookup:
    """Default external knowledge implementation using Wikipedia."""

    def lookup(
        self,
        term: str,
        context_hint: str | None = None,
        *,
        llm: Any = None,
        doc_content_chars_max: int = 1000,
        num_search_results: int = 5,
    ) -> tuple[str | None, bool]:
        if llm is None:
            raise ValueError("WikipediaLookup requires an LLM instance for relevance checks.")
        from app.core.utils.wikipedia_lookup import get_wikipedia_summary

        return get_wikipedia_summary(
            llm=llm,
            term=term,
            context_hint=context_hint,
            doc_content_chars_max=doc_content_chars_max,
            num_search_results=num_search_results,
        )


class PDFLookup:
    """External knowledge implementation using PDF files (local path or URL)."""

    def __init__(self, pdf_path_or_url: str):
        """Initialize PDF lookup with a file path or URL.

        Args:
            pdf_path_or_url: Path to a local PDF file or URL to a PDF.
        """
        self.pdf_path_or_url = pdf_path_or_url
        self._cached_text: str | None = None

    def _load_pdf_text(self) -> str:
        """Load and extract text from PDF (cached after first call)."""
        if self._cached_text is not None:
            return self._cached_text

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF support. Install with: pip install pypdf"
            )

        try:
            # Check if URL or local path
            parsed = urlparse(self.pdf_path_or_url)
            if parsed.scheme in ("http", "https"):
                # URL: download first
                import requests
                response = requests.get(self.pdf_path_or_url, timeout=30)
                response.raise_for_status()
                from io import BytesIO
                pdf_file = BytesIO(response.content)
                reader = pypdf.PdfReader(pdf_file)
            else:
                # Local file path
                if not os.path.exists(self.pdf_path_or_url):
                    raise FileNotFoundError(f"PDF file not found: {self.pdf_path_or_url}")
                reader = pypdf.PdfReader(self.pdf_path_or_url)

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            self._cached_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(self._cached_text)} characters from PDF: {self.pdf_path_or_url}")
            return self._cached_text

        except Exception as e:
            logger.error(f"Error loading PDF from {self.pdf_path_or_url}: {e}", exc_info=True)
            raise

    def _find_relevant_chunk(
        self, text: str, term: str, context_hint: str | None, llm: Any, max_chars: int = 1000
    ) -> tuple[str | None, bool]:
        """Find relevant chunk from PDF text using LLM."""
        if llm is None:
            raise ValueError("PDFLookup requires an LLM instance for relevance checks.")

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.messages import HumanMessage, SystemMessage

        # Chunk the PDF text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            logger.warning(f"No text chunks extracted from PDF: {self.pdf_path_or_url}")
            return None, False

        logger.debug(f"Split PDF into {len(chunks)} chunks for term '{term}'")

        # Use LLM to find the most relevant chunk
        term_lower = term.lower()
        context_str = f"Context: {context_hint}" if context_hint else "General context"

        system_prompt = """You are evaluating text chunks from a PDF document to find the most relevant passage for defining a specific term.
Answer with only the chunk number (0-indexed) that best defines the term, or "none" if no chunk is relevant.
Respond with a single number or "none"."""

        # Try to find chunks containing the term first
        candidate_chunks = []
        for i, chunk in enumerate(chunks):
            if term_lower in chunk.lower():
                candidate_chunks.append((i, chunk))

        if not candidate_chunks:
            # No chunk contains the term; use LLM to select best chunk
            logger.debug(f"Term '{term}' not found in any chunk; using LLM to select best match")
            chunks_text = "\n\n".join([f"[Chunk {i}]\n{chunk[:500]}" for i, chunk in enumerate(chunks[:10])])  # Limit to first 10 for prompt size
            human_prompt = f"""{context_str}
Term to define: '{term}'

Chunks from PDF:
{chunks_text}

Which chunk number (0-{min(9, len(chunks)-1)}) best defines '{term}'? Answer with the number or "none"."""

            try:
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]).content.strip().lower()

                # Parse response
                if response == "none":
                    return None, False
                try:
                    chunk_idx = int(response)
                    if 0 <= chunk_idx < len(chunks):
                        selected_chunk = chunks[chunk_idx][:max_chars]
                        logger.info(f"LLM selected chunk {chunk_idx} for term '{term}' from PDF")
                        return selected_chunk, False
                except ValueError:
                    pass
            except Exception as e:
                logger.warning(f"LLM chunk selection failed for '{term}': {e}")

            # Fallback: return first chunk
            logger.info(f"Using first chunk as fallback for term '{term}' from PDF")
            return chunks[0][:max_chars], False

        # Use the first chunk that contains the term
        selected_idx, selected_chunk = candidate_chunks[0]
        logger.info(f"Found term '{term}' in chunk {selected_idx} from PDF")
        return selected_chunk[:max_chars], False

    def lookup(
        self,
        term: str,
        context_hint: str | None = None,
        *,
        llm: Any = None,
        doc_content_chars_max: int = 1000,
    ) -> tuple[str | None, bool]:
        """Look up term in PDF document.

        Args:
            term: The term to look up.
            context_hint: Optional context (e.g. parent term).
            llm: Required LLM instance for relevance checks.
            doc_content_chars_max: Max chars for returned chunk.

        Returns:
            Tuple of (relevant text chunk or None, is_ambiguous).
            PDFs are never ambiguous (always False).
        """
        if llm is None:
            raise ValueError("PDFLookup requires an LLM instance for relevance checks.")

        try:
            text = self._load_pdf_text()
            return self._find_relevant_chunk(text, term, context_hint, llm, doc_content_chars_max)
        except Exception as e:
            logger.error(f"PDF lookup failed for term '{term}' from {self.pdf_path_or_url}: {e}", exc_info=True)
            return None, False


default_external_knowledge: ExternalKnowledgeLookup = WikipediaLookup()
