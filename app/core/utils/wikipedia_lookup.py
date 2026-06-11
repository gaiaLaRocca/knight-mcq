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

def get_wikipedia_summary(llm: ChatOpenAI, term: str, context_hint: str | None = None, doc_content_chars_max: int = 1000, num_search_results: int = 5) -> tuple[str | None, bool]:
    """
    Fetches the most relevant text chunk using LLM for relevance check.
    Checks the top `num_search_results` from wikipedia.search.
    Pipeline for each result: LLM Relevance Check -> Validation -> Fetch -> Chunk -> Select Chunk.
    Returns the first chunk containing the term, or the very first chunk as a fallback.

    Args:
        llm: The ChatOpenAI instance to use for relevance checks.
        term: The term to search for on Wikipedia.
        context_hint: Optional context (e.g., parent term) for the LLM relevance check.
        doc_content_chars_max: Max chars for the returned chunk.
        num_search_results: How many top search results to check (default 5).

    Returns:
        A tuple: (Relevant Wikipedia text chunk or None, boolean indicating if ambiguity was detected).
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
            return None, is_ambiguous
        
        logger.debug(f"[wikipedia] search for '{term}' yielded {len(search_results)} candidates: {search_results}")

        for i, page_title_guess in enumerate(search_results):
            logger.debug(f"Attempting candidate {i+1}/{len(search_results)}: '{page_title_guess}'")
            validated_title = None # Reset for each candidate
            
            # 2. LLM Relevance Check for this candidate
            if not _is_title_relevant_llm(llm, term, page_title_guess, context_hint):
                logger.debug(f"LLM relevance check failed for candidate '{page_title_guess}'. Skipping.")
                continue # Try next candidate
            logger.debug(f"LLM relevance check passed for candidate '{page_title_guess}'")

            # 3. Validate Title for this relevant candidate
            try:
                validated_page = wikipedia.page(page_title_guess, auto_suggest=False)
                validated_title = validated_page.title
                logger.debug(f"[wikipedia] validation successful. Canonical title: '{validated_title}'")
            except wikipedia.exceptions.DisambiguationError as e:
                if i == 0:
                    logger.warning(f"[wikipedia] validation failed: Top search result '{page_title_guess}' for term '{term}' is ambiguous. Stopping lookup. Options: {e.options[:5]}...")
                    is_ambiguous = True # Set flag
                    return None, is_ambiguous # Return None, True
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
            # 4. Fetch Full Text via wikipedia-api
            api_page = wiki_api.page(validated_title)
            if not api_page.exists():
                logger.warning(f"[wikipedia-api] page '{validated_title}' exists according to wikipedia.page but not wikipedia-api. Skipping candidate.")
                continue # Try next candidate
            content = api_page.text
            if not content or not content.strip():
                 logger.warning(f"Wikipedia text was empty for '{validated_title}'. Skipping candidate.")
                 continue # Try next candidate
            logger.debug(f"Fetched full text ({len(content)} chars) for '{validated_title}'")

            # 5. Chunk the Text
            chunks = text_splitter.split_text(content)
            if not chunks:
                 logger.warning(f"Text splitting yielded no chunks for '{validated_title}'. Skipping candidate.")
                 continue # Try next candidate
            logger.debug(f"Split text into {len(chunks)} chunks.")

            # 6. Select First Relevant Chunk (or fallback to first chunk)
            relevant_chunk = None
            found_exact_match = False
            search_term_lower = term.lower()
            for i_chunk, chunk in enumerate(chunks):
                if search_term_lower in chunk.lower():
                    relevant_chunk = chunk.strip()
                    found_exact_match = True
                    logger.info(f"Success! Found chunk #{i_chunk+1} containing term '{term}' in page '{validated_title}' (from candidate '{page_title_guess}')")
                    logger.debug(f"Relevant chunk snippet ({len(relevant_chunk)} chars): \"{relevant_chunk[:200]}...\"")
                    return relevant_chunk[:doc_content_chars_max], is_ambiguous # Return chunk, False
            
            # Fallback: If no chunk contained the exact term, return the first chunk
            if not found_exact_match and chunks:
                relevant_chunk = chunks[0].strip()
                logger.info(f"Term '{term}' not found in any chunk of '{validated_title}'. Falling back to first chunk.")
                logger.debug(f"Fallback chunk snippet ({len(relevant_chunk)} chars): \"{relevant_chunk[:200]}...\"")
                return relevant_chunk[:doc_content_chars_max], is_ambiguous # Return chunk, False

            # If we reach here for a candidate, it means chunking likely failed or produced empty chunks somehow
            logger.debug(f"No suitable chunk found for page '{validated_title}'. Trying next candidate.")

        logger.warning(f"Checked {len(search_results)} candidates for term '{term}', but found no relevant, valid page with usable chunks.")
        return None, is_ambiguous # Return None, False (or True if ambiguity stopped earlier)

    except requests.exceptions.RequestException as e:
        logger.error(f"Network/SSL error during Wikipedia lookup for '{term}': {e}")
        return None, False # Return None, False on network error
    except Exception as e:
        logger.error(f"Unexpected error during Wikipedia lookup for '{term}' (last guess: {page_title_guess}, last validated: {validated_title}): {e}", exc_info=True)
        return None, False # Return None, False on other errors

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
