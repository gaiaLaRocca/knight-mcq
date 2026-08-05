import logging
import wikipedia # For search and validation
import wikipediaapi # For fetching page text
import requests # For catching network errors
import re # For splitting words
from langchain.text_splitter import RecursiveCharacterTextSplitter # For chunking
from langchain_openai import ChatOpenAI # To type hint the LLM
from langchain_core.messages import HumanMessage, SystemMessage # For LLM prompt

logger = logging.getLogger(__name__)

# Shared User-Agent. Wikimedia rejects requests with generic/default User-Agents
# (returning an HTML error page that breaks JSON parsing), so we must set a
# descriptive one with a real contact on BOTH wikipedia and wikipedia-api.
WIKI_USER_AGENT = 'KNIGHT/1.0 (Contact: gaialr2001@gmail.com)'

# The `wikipedia` library (used for search/page) keeps its own default User-Agent,
# so set it explicitly. Also throttle to stay within Wikimedia rate limits.
try:
    wikipedia.set_user_agent(WIKI_USER_AGENT)
except AttributeError:
    # Older/newer variants may not expose set_user_agent; patch the module global.
    wikipedia.wikipedia.USER_AGENT = WIKI_USER_AGENT
wikipedia.set_rate_limiting(True)

# Initialize wikipedia-api object
wiki_api = wikipediaapi.Wikipedia(user_agent=WIKI_USER_AGENT, language='en')

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, length_function=len,
)

# Optional candidate-page scorers for disambiguation, injected by the thesis runner (kept
# out of the fork so it stays pinnable). See docs/phase1_kb_quality_plan.md step 5b.3.
#
# Selection is a two-stage, lexicographic rule, because "the right page" is two separate
# questions that must not be summed into one score:
#   1. entity_relevance_scorer(term, validated_title) -> float
#      "does this page talk about the ENTITY?" — a necessary condition. Scored on the
#      canonical title, not the intro: a wrong page's intro often mentions the entity in
#      passing (the Golden Gate Bridge intro names Leon Moisseiff), its title does not.
#      Keeps every candidate within PAGE_RANK_DELTA of the best — a band, not an argmax,
#      because the correct sense is not always the closest string ('Treasure Island' the
#      novel beats 'Treasure Island, San Francisco' on the title alone).
#   2. topic_relevance_scorer(text) -> float
#      "which SENSE of the entity is the right one?" — the tie-break inside that band,
#      cosine of the page intro against the topic anchor.
# Scoring only on the topic (the previous behaviour) has no entity term at all, so any
# term reachable from the topic collapses onto the topic's own page.
#
# When topic_relevance_scorer is None the legacy self-judging LLM title check runs instead.
# When only the entity scorer is None the band is every validated candidate (the previous
# topic-only ranking), so both injections are backward compatible.
topic_relevance_scorer = None
entity_relevance_scorer = None

# Width of the entity-relevance band, in cosine units below the best candidate. Calibrated
# on the golden_gate_bridge lookups (all-MiniLM-L6-v2 over canonical titles): every
# wrong-page failure observed in test_6 sits far below it (Golden Gate Bridge scores 0.090
# for 'leon moisseiff', the Bay Bridge 0.716 for 'san francisco bay'), while the genuine
# sense ambiguities that need the topic tie-break sit just inside it ('Treasure Island,
# San Francisco' 0.882 vs 1.000; 'Joseph Strauss (engineer)' 0.864 vs 1.000).
PAGE_RANK_DELTA = 0.15

def get_wikipedia_chunks(llm: ChatOpenAI, term: str, context_hint: str | None = None, topic: str | None = None, doc_content_chars_max: int = 1000, num_search_results: int = 5, per_section: bool = False) -> tuple[list[str], bool, str | None]:
    """
    Fetches the relevant text chunks for `term` from the best of the top
    `num_search_results` wikipedia.search candidates.

    Legacy pipeline (no scorers injected), per candidate, first match wins:
        LLM Relevance Check -> Validation -> Fetch -> Chunk -> Select Chunks.
    Ranked pipeline (scorers injected, see the module globals):
        Validation (all candidates) -> entity-relevance band -> Fetch/Chunk/Select the
        in-band ones -> topic-relevance tie-break.
    Returns the topic chunk and the entity chunk of the winning page (see `_select_chunks`),
    or its lead plus one chunk per body section when `per_section` is set.

    Args:
        llm: The ChatOpenAI instance to use for relevance checks.
        term: The term to search for on Wikipedia.
        context_hint: Optional context (e.g., parent term) for the LLM relevance check.
        topic: Optional overall topic; when set, a chunk mentioning it is added so the
            entity's link to the topic is retained (the two-chunk rule). When None, only
            the entity chunk is returned (legacy single-chunk behaviour).
        doc_content_chars_max: Max chars for each returned chunk.
        num_search_results: How many top search results to check (default 5).
        per_section: Return the lead plus the opening chunk of every body section instead
            of the two-chunk selection. For the topic entity only (see `_section_chunks`).

    Returns:
        A tuple: (list of Wikipedia text chunks (possibly empty), boolean indicating if
        ambiguity was detected, canonical title of the winning page or None). The title is
        returned so callers can record the page the text actually came from — the search
        term is not it, and building a URL from the term yields a dead link.
    """
    logger.info(f"Performing Wikipedia lookup for term: '{term}'") # Log lookup attempt
    page_title_guess = None
    validated_title = None
    is_ambiguous = False # Flag for ambiguity
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(term, results=num_search_results)
        if not search_results:
            logger.warning(f"[wikipedia] search found no results for term: {term}")
            return [], is_ambiguous, None

        logger.debug(f"[wikipedia] search for '{term}' yielded {len(search_results)} candidates: {search_results}")

        scorer = topic_relevance_scorer  # thesis-injected; None => legacy LLM-check path

        # --- Stage 1: validate the candidates (titles only, no full-text fetch yet) ---
        # The ranked path needs every candidate's canonical title before it can score any
        # of them, so validation is separated from the fetch. This also makes the ranked
        # path *cheaper* than before: only the in-band candidates are fetched and chunked.
        validated_candidates = []  # (validated_title, page_title_guess)
        for i, page_title_guess in enumerate(search_results):
            logger.debug(f"Attempting candidate {i+1}/{len(search_results)}: '{page_title_guess}'")
            validated_title = None # Reset for each candidate

            # Legacy path uses the self-judging LLM title check as the filter. The ranked
            # path (scorer set) skips it and instead selects among the validated pages by
            # entity relevance then topic relevance below (step 5b.3), removing that
            # circularity and the per-candidate LLM call.
            if scorer is None and not _is_title_relevant_llm(llm, term, page_title_guess, context_hint):
                logger.debug(f"LLM relevance check failed for candidate '{page_title_guess}'. Skipping.")
                continue # Try next candidate

            # Validate Title for this candidate
            try:
                validated_page = wikipedia.page(page_title_guess, auto_suggest=False)
                validated_title = validated_page.title
                logger.debug(f"[wikipedia] validation successful. Canonical title: '{validated_title}'")
            except wikipedia.exceptions.DisambiguationError as e:
                if i == 0:
                    logger.warning(f"[wikipedia] validation failed: Top search result '{page_title_guess}' for term '{term}' is ambiguous. Stopping lookup. Options: {e.options[:5]}...")
                    is_ambiguous = True # Set flag
                    return [], is_ambiguous, None # Return [], True, None
                else:
                    logger.debug(f"[wikipedia] validation failed: Candidate '{page_title_guess}' is ambiguous. Skipping candidate.")
                    continue # Try next candidate
            except wikipedia.exceptions.PageError as e:
                logger.debug(f"[wikipedia] validation failed: Page '{page_title_guess}' does not exist. Skipping candidate.")
                continue # Try next candidate
            except Exception as e_val:
                 logger.warning(f"Unexpected error during wikipedia.page validation for '{page_title_guess}': {e_val}. Skipping candidate.")
                 continue # Try next candidate

            if scorer is None:
                # Legacy: the first valid, LLM-approved candidate that yields chunks wins,
                # so fetch it right away and stop as soon as one does.
                selected, _lead = _fetch_and_select(validated_title, topic, doc_content_chars_max, per_section)
                if not selected:
                    continue # Try next candidate
                logger.info(f"Success! Selected {len(selected)} chunk(s) for term '{term}' in page '{validated_title}' (from candidate '{page_title_guess}')")
                logger.debug(f"First selected chunk snippet: \"{selected[0][:200]}...\"")
                return selected, is_ambiguous, validated_title

            validated_candidates.append((validated_title, page_title_guess))

        if scorer is not None and validated_candidates:
            # --- Stage 2: entity-relevance band (does the page talk about the entity?) ---
            band = _entity_relevance_band(term, [t for t, _ in validated_candidates])

            # --- Stage 3: topic tie-break inside the band (which sense of the entity?) ---
            ranked = []  # (topic_score, selected, validated_title)
            for validated_title in band:
                selected, lead = _fetch_and_select(validated_title, topic, doc_content_chars_max, per_section)
                if not selected:
                    continue # Try next in-band candidate
                # Score the page's own intro — the first chunk of the full split, NOT the
                # first *selected* chunk, which is the topic chunk and would make every
                # page score as its most on-topic paragraph.
                try:
                    score = scorer(lead)
                except Exception as e_score:
                    logger.warning(f"[wikipedia] topic-relevance scorer failed for '{validated_title}': {e_score}. Scoring -inf.")
                    score = float("-inf")
                logger.debug(f"[wikipedia] in-band candidate '{validated_title}' topic-relevance score={score:.4f}")
                ranked.append((score, selected, validated_title))

            if ranked:
                best_score, best_selected, best_title = max(ranked, key=lambda c: c[0])
                logger.info(f"Selected page '{best_title}' for term '{term}': entity band of {len(band)}/{len(validated_candidates)} candidate(s), topic-relevance={best_score:.4f}.")
                return best_selected, is_ambiguous, best_title

        logger.warning(f"Checked {len(search_results)} candidates for term '{term}', but found no relevant, valid page with usable chunks.")
        return [], is_ambiguous, None # Return [], False (or True if ambiguity stopped earlier)

    except requests.exceptions.RequestException as e:
        logger.error(f"Network/SSL error during Wikipedia lookup for '{term}': {e}")
        return [], False, None # Return [], False on network error
    except Exception as e:
        logger.error(f"Unexpected error during Wikipedia lookup for '{term}' (last guess: {page_title_guess}, last validated: {validated_title}): {e}", exc_info=True)
        return [], False, None # Return [], False on other errors

# Wikipedia's standard appendix sections (WP:LAYOUT). They carry no article prose — only
# citations, bibliographies and link lists — yet they are dense in the article's own subject
# terms, so a topic-relevance argmax over the raw page text lands on them: on test_6 Joseph
# Strauss's "References / Further reading" scored 0.562 against the topic, beating the best
# real paragraph of his page (0.537). No score rule can separate the two, because the
# bibliography genuinely *is* the densest occurrence of the topic string. The exclusion has to
# be structural, and it has to happen **before** chunking: dropping bad chunks afterwards
# leaves the chunk that straddles the last body section and the first appendix heading.
_APPENDIX_SECTIONS = frozenset({
    "references", "further reading", "external links", "see also", "notes", "citations",
    "bibliography", "sources", "footnotes", "works cited", "gallery",
})


def _body_sections(api_page):
    """The page's top-level sections minus the appendices (see `_APPENDIX_SECTIONS`)."""
    return [s for s in api_page.sections if s.title.strip().lower() not in _APPENDIX_SECTIONS]


def _body_text(api_page) -> str:
    """The article body: the lead plus every non-appendix top-level section."""
    parts = [api_page.summary] + [s.full_text() for s in _body_sections(api_page)]
    return "\n".join(p for p in parts if p and p.strip())


def _section_chunks(api_page, max_chars: int) -> list[str]:
    """The lead plus the opening chunk of each body section (the `per_section` mode).

    Used for the topic entity only. The topic node is not one entity among many — it is the
    concept Phase 2 measures — so representing it with the same two chunks as `marin county`
    under-grounds precisely the thing under test. Taking the *first* chunk of each section
    rather than a cosine argmax is deliberate: inside the topic's own page every chunk scores
    high against the topic, so the ranking carries no information, whereas a section's opening
    is by editorial convention the paragraph that introduces it.

    The lead is taken **whole and untruncated** — neither `split(lead)[0]` nor
    `lead[:max_chars]`. Both cuts land in the wrong place on the Golden Gate Bridge, whose
    1614-character lead carries its most interrogable facts at the very end: the splitter
    stops at 657, and `max_chars` at 1000, while `May 27, 1937` sits at offset 1049,
    `4,200 feet` at 1552 and `746 feet` at 1597. Truncating would drop the opening date, the
    main span and the height from a knowledge base about the bridge.

    Exempting the topic from `max_chars` is safe precisely because it is the topic: the cap
    exists so the gate scores the same text the KB carries (minilm truncates past ~1250
    characters), and the topic is never scored for pruning — it is kept by construction. Every
    other entity keeps the cap. Cost here is ~150 tokens. Sections are split, since only their
    first chunk is wanted.
    """
    selected: list[str] = []
    lead = (api_page.summary or "").strip()
    if lead:
        selected.append(lead)
    for section in _body_sections(api_page):
        text = section.full_text()
        if not text or not text.strip():
            continue
        pieces = text_splitter.split_text(text)
        if not pieces:
            continue
        chunk = pieces[0].strip()[:max_chars]
        if chunk and not _covered_by(chunk, selected):
            selected.append(chunk)
    return selected


def _covered_by(chunk: str, selected: list[str]) -> bool:
    """True if `chunk` adds nothing to `selected` — its whole text is already inside one.

    Plain equality is not enough: the same lead reaches the selection by two routes — whole
    (`summary[:max_chars]`) and as the splitter's opening piece of the body — and the two
    differ only in length, so an equality check keeps both. Deliberately asymmetric: a chunk
    that *contains* an already-selected one is richer, not redundant, and must not be dropped
    on that basis (callers drop the subsumed one instead).
    """
    return any(chunk in s for s in selected)


def _fetch_and_select(validated_title: str, topic: str | None, max_chars: int,
                      per_section: bool = False) -> tuple[list[str], str]:
    """Fetch a validated page, chunk its body, and select its chunks. ([], "") on failure.

    Returns `(selected, lead)`: the chunks kept for this page and the page's lead section.
    The lead is returned separately because `selected[0]` is the topic chunk — the page's most
    on-topic paragraph — which is the wrong text to judge *which page this is* with. Only the
    article body is chunked; appendices never become chunks at all.
    """
    api_page = wiki_api.page(validated_title)
    if not api_page.exists():
        logger.warning(f"[wikipedia-api] page '{validated_title}' exists according to wikipedia.page but not wikipedia-api. Skipping candidate.")
        return [], ""
    body = _body_text(api_page)
    if not body.strip():
        logger.warning(f"Wikipedia body text was empty for '{validated_title}'. Skipping candidate.")
        return [], ""
    logger.debug(f"Fetched body text ({len(body)} chars, {len(_body_sections(api_page))} sections) for '{validated_title}'")

    # The entity chunk is the article's **opening window**: the first `max_chars` of the body,
    # which always begins at the lead and runs on into the article when the lead is short.
    #
    # Not `split(body)[0]`, which stops at the first paragraph boundary that fits — 657 of the
    # Golden Gate Bridge's 1614-character lead, cutting the span and height figures. And not
    # `summary[:max_chars]` either: Joseph Strauss's lead is 224 characters, so a lead-only
    # rule would spend a quarter of the window and leave the rest empty, dropping the
    # biography — including the Cincinnati birthplace that the two-chunk design exists to keep
    # answerable ("where was the designer of the Golden Gate Bridge born?"). The window gives
    # the definition first and fills the remaining budget with the article's own continuation.
    opening = body.strip()[:max_chars]

    if per_section:
        selected = _section_chunks(api_page, max_chars)
    else:
        chunks = text_splitter.split_text(body)
        if not chunks:
            logger.warning(f"Text splitting yielded no chunks for '{validated_title}'. Skipping candidate.")
            return [], ""
        logger.debug(f"Split body into {len(chunks)} chunks.")
        selected = _select_chunks(chunks, topic=topic, max_chars=max_chars, lead=opening)

    if not selected:
        logger.debug(f"No suitable chunk found for page '{validated_title}'. Trying next candidate.")
        return [], ""
    return selected, opening or selected[0]


def _entity_relevance_band(term: str, titles: list[str]) -> list[str]:
    """Keep the candidate titles that plausibly denote `term` (stage 1 of the selection).

    Scores each canonical title with the injected `entity_relevance_scorer` and keeps
    everything within `PAGE_RANK_DELTA` of the best. A band rather than an argmax: the
    closest title is not always the right sense, and the topic tie-break that follows
    needs the alternatives to still be on the table. Falls back to every candidate when
    no scorer is injected or scoring fails, leaving the topic-only ranking in place.
    """
    scorer = entity_relevance_scorer
    if scorer is None or not titles:
        return titles
    try:
        scores = [(t, scorer(term, t)) for t in titles]
    except Exception as e:
        logger.warning(f"[wikipedia] entity-relevance scorer failed for '{term}': {e}. Keeping all candidates.")
        return titles
    best = max(s for _, s in scores)
    band = [t for t, s in scores if s >= best - PAGE_RANK_DELTA]
    logger.debug(
        "[wikipedia] entity relevance for '%s': %s -> band %s",
        term, [f"{t}={s:.3f}" for t, s in scores], band,
    )
    return band


def _is_title_relevant_llm(llm: ChatOpenAI, term: str, title_guess: str, context_hint: str | None) -> bool:
    """Uses LLM to check if a Wikipedia title is relevant for defining a term in context.
    Now uses System + Human prompts.
    """
    if not context_hint:
        context_hint = "general knowledge"
    
    # System prompt defining the task and constraints
    system_prompt_content = ("""
You are performing a relevance classification task to evaluate whether a Wikipedia page title is an appropriate definition source for a given term within a specific context. 
You are expected to act as a domain-specific semantic filter. 
Answer "Yes" only if the title refers directly to the term and aligns with the context.
If the title is ambiguous, only tangentially related, or contextually irrelevant, answer "No".
Respond with only one word: "Yes" or "No".
""")

    # Human prompt providing the specific data for evaluation
    human_prompt_content = (
        f"Context: Information related to '{context_hint}'.\n"
        f"Term to define: '{term}'\n"
        f"Candidate Wikipedia Page Title: '{title_guess}'\n\n"
        f"Evaluate relevance and respond with only 'Yes' or 'No'."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=human_prompt_content)
        ]).content.strip().lower()
        
        logger.debug(f"LLM relevance check for '{term}' -> '{title_guess}' (context: '{context_hint}'): Response='{response}'")
        # Be strict about the expected answer
        if response == "yes":
             return True
        elif response == "no":
             return False
        else:
             logger.warning(f"LLM relevance check returned unexpected answer: '{response}'. Assuming not relevant.")
             return False
    except Exception as e:
        logger.error(f"Error during LLM relevance check for '{term}' -> '{title_guess}': {e}", exc_info=True)
        return False # Assume not relevant on error

def _select_chunks(chunks: list[str], topic: str | None, max_chars: int, lead: str = "") -> list[str]:
    """Pick up to two de-duplicated chunks from one validated page, topic chunk first.

    Implements the two-chunk rule of docs/phase1_kb_quality_plan.md (step 3,
    "Chunk selection within a kept page"):

      1. topic chunk  -- the first chunk mentioning the topic (e.g. "golden gate
         bridge"): WHY the entity matters to the topic. Skipped when no topic is
         given or no chunk mentions it (e.g. an off-topic page like `fog`).
      2. entity chunk -- the page's **lead** section: the entity's own definition.

    Multi-hop graph-path questions need both halves (e.g. "where was the designer of
    the Golden Gate Bridge born?": the topic link is in the topic chunk, the
    birthplace in the biographical lead). Each chunk is stripped and truncated
    to `max_chars`; if the topic chunk and the entity chunk are the same chunk it is
    returned once. Returns [] only when `chunks` is empty.

    The entity chunk used to be "the first chunk containing the term, else chunks[0]",
    which broke on any name the lead does not spell exactly: `leon moisseiff` first
    appears literally in the body of his page (his lead reads "Leon *Solomon*
    Moisseiff"), so the rule returned a body chunk that was also the topic chunk, the
    two collapsed, and the entity was left with no definition at all. The lead is what
    the rule always meant, and it is now safe to take it unconditionally because the
    entity-relevance band already guarantees the page is about the term.
    """
    if not chunks:
        return []
    selected: list[str] = []

    if topic:
        topic_lower = topic.lower()
        topic_chunk = next((c for c in chunks if topic_lower in c.lower()), None)
        if topic_chunk is not None:
            selected.append(topic_chunk.strip()[:max_chars])

    entity_chunk = (lead or chunks[0]).strip()[:max_chars]
    if entity_chunk:
        # When the topic chunk is merely an opening slice of the lead — which happens on
        # every page whose lead is both the most on-topic passage and longer than one chunk
        # — the lead subsumes it, so drop the slice and keep the richer text rather than the
        # other way round.
        selected = [s for s in selected if s not in entity_chunk]
        if not _covered_by(entity_chunk, selected):
            selected.append(entity_chunk)

    return selected

def get_wikipedia_summary(llm: ChatOpenAI, term: str, context_hint: str | None = None, topic: str | None = None, doc_content_chars_max: int = 1000, num_search_results: int = 5) -> tuple[str | None, bool]:
    """Backward-compatible single-string view over `get_wikipedia_chunks`.

    Joins the selected chunks into one summary block (None if none were found), so
    existing callers that expect a single text context (term-description generation)
    are unchanged. Pass `topic` to also get the topic chunk; omit it for the legacy
    entity-only chunk.
    """
    chunks, is_ambiguous, _title = get_wikipedia_chunks(
        llm=llm,
        term=term,
        context_hint=context_hint,
        topic=topic,
        doc_content_chars_max=doc_content_chars_max,
        num_search_results=num_search_results,
    )
    summary = "\n\n".join(chunks) if chunks else None
    return summary, is_ambiguous

# Update example usage expectations
if __name__ == '__main__':
    # NOTE: This test block will NOT work correctly without providing an LLM instance.
    # You would need to instantiate one here similar to how it's done in chatbot.py,
    # or run the tests through the main chatbot flow.
    logging.basicConfig(level=logging.DEBUG)
    print("\n*** NOTE: Running this file directly requires LLM configuration for relevance checks. ***\n")
    # Example call structure (won't run without llm):
    # from app.core.common.config import OPENAI_API_KEY, OPENAI_MODEL 
    # test_llm = ChatOpenAI(model=OPENAI_MODEL, base_url="...", api_key=OPENAI_API_KEY)
    # term1 = "Artificial Intelligence"
    # content1 = get_wikipedia_summary(test_llm, term1, context_hint="technology")
    # ... etc ...
    print("Direct execution skipped as LLM instance is needed.")
