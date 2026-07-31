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

# Optional topic-relevance scorer for candidate-page disambiguation, injected by the
# thesis runner (kept out of the fork so it stays pinnable). A callable str -> float
# returning cosine(embed(text), embed(topic_anchor)). When set, get_wikipedia_chunks
# ranks the validated candidate pages by this score and keeps the best, instead of
# taking the first page that passes the self-judging LLM title check (see
# docs/phase1_kb_quality_plan.md step 5b.3). When None, the legacy behaviour holds.
topic_relevance_scorer = None

def get_wikipedia_chunks(llm: ChatOpenAI, term: str, context_hint: str | None = None, topic: str | None = None, doc_content_chars_max: int = 1000, num_search_results: int = 5) -> tuple[list[str], bool]:
    """
    Fetches up to two relevant text chunks using LLM for relevance check.
    Checks the top `num_search_results` from wikipedia.search.
    Pipeline for each result: LLM Relevance Check -> Validation -> Fetch -> Chunk -> Select Chunks.
    Returns the topic chunk and the entity chunk of the first valid page (see `_select_chunks`).

    Args:
        llm: The ChatOpenAI instance to use for relevance checks.
        term: The term to search for on Wikipedia.
        context_hint: Optional context (e.g., parent term) for the LLM relevance check.
        topic: Optional overall topic; when set, a chunk mentioning it is added so the
            entity's link to the topic is retained (the two-chunk rule). When None, only
            the entity chunk is returned (legacy single-chunk behaviour).
        doc_content_chars_max: Max chars for each returned chunk.
        num_search_results: How many top search results to check (default 5).

    Returns:
        A tuple: (list of up to two Wikipedia text chunks (possibly empty),
        boolean indicating if ambiguity was detected).
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
            return [], is_ambiguous

        logger.debug(f"[wikipedia] search for '{term}' yielded {len(search_results)} candidates: {search_results}")

        scorer = topic_relevance_scorer  # thesis-injected; None => legacy LLM-check path
        ranked = []  # (score, selected, validated_title), collected when scorer is set

        for i, page_title_guess in enumerate(search_results):
            logger.debug(f"Attempting candidate {i+1}/{len(search_results)}: '{page_title_guess}'")
            validated_title = None # Reset for each candidate

            # Legacy path uses the self-judging LLM title check as the filter. The ranked
            # path (scorer set) skips it and instead ranks the validated pages by topic
            # cosine below (step 5b.3), removing that circularity and the per-candidate
            # LLM call.
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
                    return [], is_ambiguous # Return [], True
                else:
                    logger.debug(f"[wikipedia] validation failed: Candidate '{page_title_guess}' is ambiguous. Skipping candidate.")
                    continue # Try next candidate
            except wikipedia.exceptions.PageError as e:
                logger.debug(f"[wikipedia] validation failed: Page '{page_title_guess}' does not exist. Skipping candidate.")
                continue # Try next candidate
            except Exception as e_val:
                 logger.warning(f"Unexpected error during wikipedia.page validation for '{page_title_guess}': {e_val}. Skipping candidate.")
                 continue # Try next candidate

            # If validation succeeded, proceed to fetch and chunk
            # Fetch Full Text via wikipedia-api
            api_page = wiki_api.page(validated_title)
            if not api_page.exists():
                logger.warning(f"[wikipedia-api] page '{validated_title}' exists according to wikipedia.page but not wikipedia-api. Skipping candidate.")
                continue # Try next candidate
            content = api_page.text
            if not content or not content.strip():
                 logger.warning(f"Wikipedia text was empty for '{validated_title}'. Skipping candidate.")
                 continue # Try next candidate
            logger.debug(f"Fetched full text ({len(content)} chars) for '{validated_title}'")

            # Chunk the Text
            chunks = text_splitter.split_text(content)
            if not chunks:
                 logger.warning(f"Text splitting yielded no chunks for '{validated_title}'. Skipping candidate.")
                 continue # Try next candidate
            logger.debug(f"Split text into {len(chunks)} chunks.")

            # Select up to two chunks: the topic chunk and the entity chunk.
            selected = _select_chunks(chunks, term=term, topic=topic, max_chars=doc_content_chars_max)
            if not selected:
                logger.debug(f"No suitable chunk found for page '{validated_title}'. Trying next candidate.")
                continue # Try next candidate

            if scorer is None:
                # Legacy: the first valid, LLM-approved candidate wins.
                logger.info(f"Success! Selected {len(selected)} chunk(s) for term '{term}' in page '{validated_title}' (from candidate '{page_title_guess}')")
                logger.debug(f"First selected chunk snippet: \"{selected[0][:200]}...\"")
                return selected, is_ambiguous

            # Ranked path: score the page's own intro (chunks[0]) against the topic anchor
            # and keep the candidate for the argmax below.
            try:
                score = scorer(chunks[0])
            except Exception as e_score:
                logger.warning(f"[wikipedia] topic-relevance scorer failed for '{validated_title}': {e_score}. Scoring -inf.")
                score = float("-inf")
            logger.debug(f"[wikipedia] candidate '{validated_title}' topic-relevance score={score:.4f}")
            ranked.append((score, selected, validated_title))

        if scorer is not None and ranked:
            best_score, best_selected, best_title = max(ranked, key=lambda c: c[0])
            logger.info(f"Selected page '{best_title}' for term '{term}' by topic-relevance (score={best_score:.4f}) among {len(ranked)} candidate(s).")
            return best_selected, is_ambiguous

        logger.warning(f"Checked {len(search_results)} candidates for term '{term}', but found no relevant, valid page with usable chunks.")
        return [], is_ambiguous # Return [], False (or True if ambiguity stopped earlier)

    except requests.exceptions.RequestException as e:
        logger.error(f"Network/SSL error during Wikipedia lookup for '{term}': {e}")
        return [], False # Return [], False on network error
    except Exception as e:
        logger.error(f"Unexpected error during Wikipedia lookup for '{term}' (last guess: {page_title_guess}, last validated: {validated_title}): {e}", exc_info=True)
        return [], False # Return [], False on other errors

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

def _select_chunks(chunks: list[str], term: str, topic: str | None, max_chars: int) -> list[str]:
    """Pick up to two de-duplicated chunks from one validated page, topic chunk first.

    Implements the two-chunk rule of docs/phase1_kb_quality_plan.md (step 3,
    "Chunk selection within a kept page"):

      1. topic chunk  -- the first chunk mentioning the topic (e.g. "golden gate
         bridge"): WHY the entity matters to the topic. Skipped when no topic is
         given or no chunk mentions it (e.g. an off-topic page like `fog`).
      2. entity chunk -- the first chunk mentioning the term, else the first chunk:
         the entity's own intro / definition.

    Multi-hop graph-path questions need both halves (e.g. "where was the designer of
    the Golden Gate Bridge born?": the topic link is in the topic chunk, the
    birthplace in the biographical intro chunk). Each chunk is stripped and truncated
    to `max_chars`; if the topic chunk and the entity chunk are the same chunk it is
    returned once. Returns [] only when `chunks` is empty.
    """
    if not chunks:
        return []
    selected: list[str] = []

    if topic:
        topic_lower = topic.lower()
        topic_chunk = next((c for c in chunks if topic_lower in c.lower()), None)
        if topic_chunk is not None:
            selected.append(topic_chunk.strip()[:max_chars])

    term_lower = term.lower()
    entity_chunk = next((c for c in chunks if term_lower in c.lower()), chunks[0])
    entity_chunk = entity_chunk.strip()[:max_chars]
    if entity_chunk not in selected:
        selected.append(entity_chunk)

    return selected

def get_wikipedia_summary(llm: ChatOpenAI, term: str, context_hint: str | None = None, topic: str | None = None, doc_content_chars_max: int = 1000, num_search_results: int = 5) -> tuple[str | None, bool]:
    """Backward-compatible single-string view over `get_wikipedia_chunks`.

    Joins the selected chunks into one summary block (None if none were found), so
    existing callers that expect a single text context (term-description generation)
    are unchanged. Pass `topic` to also get the topic chunk; omit it for the legacy
    entity-only chunk.
    """
    chunks, is_ambiguous = get_wikipedia_chunks(
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
