import logging
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.common.config import (
    OPENAI_API_KEY,
    DEFAULT_NO_DESCRIPTION,
    DEFAULT_ERROR_DESCRIPTION,
    OPENAI_MODEL,
)
from app.core.utils.external_knowledge import (
    ExternalKnowledgeLookup,
    default_external_knowledge,
)

# Use the named logger configured in chatbot.py
logger = logging.getLogger("gpt_agent") 

# NOTE: LLM instance is now passed into generate_term_description
# llm = ChatOpenAI(...) # Remove global instance if only used here

processed_descriptions = set()  # Track processed descriptions globally

# Replaces KNIGHT's 8-point template (definition, domains of use, subfields, mechanisms,
# applications, case studies, related terms, current research). See
# thesis-concept-robustness/docs/phase1_kb_quality_plan.md, "Node descriptions anchored to the
# Wikipedia context". The template suits scientific concepts; for an entity such as Leon
# Moisseiff or the Pacific Ocean it made the model fill sections from memory: on test_7 every
# description ran 940-1,157 words from about 290 words of Wikipedia context, and questions
# built on them had answers the KB cannot support. Every clause below has a reason:
# - prose: nothing downstream parses the sections, and prose gives coreference an anaphoric
#   chain to resolve, as the seed essay does;
# - no preamble, definition first: the QA validator reads only the first 150 characters of a
#   description, and on test_7 all 22 opened with "Okay, here's a detailed explanation...";
# - about 300 words: parity with the context, not brevity. Questions are built from the
#   descriptions and answered from the KB, so a note shorter than its source drops facts the
#   chunks would support;
# - stay close to the material: soft for now; a strict version waits for test_8's evidence.
DESCRIPTION_SYSTEM_PROMPT = """You are a subject-matter expert writing short reference notes. Explain the term the user gives you in continuous prose: no lists, no headings, no markdown, and no preamble. Begin directly with a one-sentence definition of the term, then continue with what matters most about it.

Write about 300 words. The reference material you are given is the primary source: stay close to what it says. Prefer concrete, checkable facts (names, places, dates, measurements, roles) over general commentary, and do not pad the note with background the material does not support. If the material is thin, write a shorter note instead of filling it out."""

# The chatter a small model puts before the content despite being told not to. What marks it
# is not the opening word but that it talks about the answer instead of the entity: an
# optional interjection followed by an announcement ("Okay, here's...", "Okay, let's delve
# into...", "Here is a note on..."). All 22 test_7 descriptions match. "Okay, Leon Moisseiff
# was an engineer..." does not, and neither do Okayama or Hereford: facts are never dropped.
_PREAMBLE = re.compile(
    r"^(?:(?:okay|ok|sure|certainly|of course|alright|absolutely)[,.!]?\s+)?"
    r"(?:here's|here’s|here is|let's|let’s|let me|i'll|i’ll|i will|below is)\b",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    """Drop the announcement a small model puts before the content, and nothing else.

    The prompt forbids the preamble; this enforces it, because the first 150 characters are
    all the QA validator reads. Losing a fact costs more than keeping a preamble, so only text
    that is certainly the announcement is removed, and the rest of the description is always
    kept. When the first line announces the answer (see `_PREAMBLE`):

    - with a colon, everything up to the colon goes and what follows it stays, on the same
      line or below ("...the requested structure: **Term:** US Route 101...");
    - without one, the line goes only if it is a single sentence with text after it;
    - otherwise nothing is removed: "Okay, here's a note on X. X was an engineer..." keeps its
      facts, and so does a preamble with "U.S." in it, whose sentence end cannot be told
      from the abbreviation (an automatic sentence splitter cuts right after "U.S.").
    """
    text = text.strip()
    first_line = text.split("\n", 1)[0]
    match = _PREAMBLE.match(first_line)
    if not match:
        return text
    if ":" in first_line:
        cut = text.index(":") + 1
    elif not re.search(r"[.!?]\s", first_line[match.end():].rstrip()):
        cut = len(first_line)
    else:
        return text
    rest = text[cut:].strip()
    if not rest:
        return text
    logger.info(f"Dropped description preamble: {text[:cut].strip()[:200]!r}")
    return rest

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def generate_term_description(
    llm: ChatOpenAI,
    term: str,
    parent_term: str | None = None,
    source_context_text: str | None = None,
    external_lookup: ExternalKnowledgeLookup | None = None,
) -> tuple[str, bool]:
    """
    Generate a term description using the provided LLM instance.
    Uses external knowledge (default: Wikipedia) as context if available and unambiguous.
    Uses source_context_text for LLM prompt context otherwise.
    Returns the description string and a boolean indicating if external knowledge context was used.
    """
    if external_lookup is None:
        external_lookup = default_external_knowledge
    wikipedia_context_used = False
    try:
        base_prompt = DESCRIPTION_SYSTEM_PROMPT

        # Call external knowledge lookup (default: Wikipedia), expecting tuple (summary, is_ambiguous)
        wikipedia_summary, is_ambiguous = external_lookup.lookup(
            term=term,
            context_hint=parent_term or source_context_text,
            llm=llm,
        )

        # Use Wikipedia context only if found and not ambiguous
        if wikipedia_summary and not is_ambiguous: # <<< This `if` must be indented same level as base_prompt definition
            wikipedia_context_used = True
            logger.info(f"Found unambiguous Wikipedia context for '{term}'. Using it for LLM description generation.")

            # Construct Human Prompt for Wikipedia context case
            task_instruction_wiki = f"Explain the term: '{term}'."
            context_instruction_wiki = f"""Use the following Wikipedia context as the primary source for your explanation:
--- Wikipedia Context ---
{wikipedia_summary}
--- End Wikipedia Context ---"""
            # Kept from KNIGHT: it is what yields the edge back to the parent.
            parent_hint_wiki = f"Also explain how it relates to '{parent_term}'." if parent_term else ""
            human_prompt_content_wiki = f"{task_instruction_wiki}\n\n{context_instruction_wiki}\n\n{parent_hint_wiki}".strip()

            # Call the LLM with the standard System prompt and the specific Human prompt
            logger.debug(f"Generating description for '{term}' using System Prompt + Human Prompt with Wiki context. Human: {human_prompt_content_wiki[:400]}...")
            response = llm.invoke([
                SystemMessage(content=base_prompt), # Use the common system prompt
                HumanMessage(content=human_prompt_content_wiki)
            ]).content

        else:

            # Handle ambiguous or no Wikipedia results
            if is_ambiguous:
                 logger.warning(f"Wikipedia result for '{term}' is ambiguous. Falling back to LLM with source context.")
            else: # No summary found
                 logger.info(f"No suitable Wikipedia context found for '{term}'. Generating description using LLM and source context if available.")

            # *** START: Modified Fallback Prompt Handling ***
            # Task Specific Instruction (for no-wiki case). Same system prompt: such a node is
            # pruned as ungrounded before QA, so only the cost of a long answer is at stake.
            task_instruction = f"Explain the term: '{term}'."

            # Add context if available (parent term or source text)
            context_hint = None
            if parent_term:
                context_hint = f"Also explain how it relates to '{parent_term}'."
            if source_context_text:
                context_hint = (context_hint + "\n" if context_hint else "") + f"Additional context from source text: {source_context_text}"

            if context_hint:
                task_instruction += f"\n{context_hint}"

            # Combine Base Prompt (System) and Task Instruction (Human)
            system_prompt_content = base_prompt # Use the common system prompt
            human_prompt_content = task_instruction
            
            # Invoke with separate System and Human messages
            logger.debug(f"Generating description for '{term}' without Wikipedia context (System + Human). System Prompt: {system_prompt_content[:200]}... Human Prompt: {human_prompt_content[:300]}...")
            response = llm.invoke([
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=human_prompt_content)
            ]).content
            # *** END: Modified Fallback Prompt Handling ***
        
        description = _strip_preamble(response)
        if description:
            logger.info(f"Generated description for term '{term}' (Wikipedia context: {'Yes' if wikipedia_context_used else 'No'})")
            if wikipedia_context_used:
                # The test_8 measurement: description length against the context it was
                # written from (about 3.5x on test_7; the target is about 1x).
                desc_words, ctx_words = len(description.split()), len(wikipedia_summary.split())
                logger.info(f"Description length for '{term}': {desc_words} words from {ctx_words} words of Wikipedia context ({desc_words / max(ctx_words, 1):.1f}x).")
            return description, wikipedia_context_used
        else:
            logger.warning(f"No definition returned by LLM for term: '{term}'")
            # Return default description and False for the flag
            return DEFAULT_NO_DESCRIPTION, False 
            
    except Exception as e:
        logger.error(f"Error generating description for term '{term}': {e}")
        # Raise the exception for retry logic, but if retries fail, this won't return normally.
        # If we needed a value on final failure, we'd handle it differently, but retry handles it.
        raise 

def save_term_description(conn, term, description, wikipedia_context_used: bool):
    """
    Save the term description and the fact-checking flag to the Neo4j database.
    """
    global processed_descriptions
    description_clean = description.replace("'", "\\'").replace("\n", " ").strip()
    
    # Determine the string value for the flag
    fact_checked_value = "Yes" if wikipedia_context_used else "No"
    
    # Check if this specific description has already been processed to avoid redundant writes
    # (Note: This doesn't prevent overwriting an old description with a new one)
    if description_clean in processed_descriptions:
        logger.debug(f"Description '{description_clean[:50]}...' already processed. Skipping save for term '{term}'.")
        # If description is identical, maybe update flag? For now, skip.
        return

    # Use MERGE to find or create the term, then SET description and flag
    # This overwrites existing description and flag if the term exists
    query = """
    MERGE (t:Term {name: $term})
    SET t.description = $description, t.wiki_fact_checked = $fact_checked
    """
    try:
        logger.debug(f"Attempting to save description for term '{term}' (Wiki Fact Checked: {fact_checked_value}).")
        conn.execute_write(query, parameters={"term": term, "description": description_clean, "fact_checked": fact_checked_value})
        processed_descriptions.add(description_clean)
        logger.info(f"Description for term '{term}' saved successfully (Wiki Fact Checked: {fact_checked_value}).")
    except Exception as e:
        logger.error(f"Error saving description for term '{term}': {e}")
        raise

def query_term_description(conn, term):
    """
    Query the description of a term from the Neo4j database.
    """
    query = """
    MATCH (t:Term {name: $term})
    RETURN t.description AS description
    """
    try:
        results = conn.query(query, parameters={"term": term})
        if results:
            description = results[0]["description"]
            logger.debug(f"Retrieved description for term '{term}': {description}")
            return description
        else:
            logger.debug(f"No description found for term '{term}'.")
    except Exception as e:
        logger.error(f"Error querying term '{term}': {e}")
    return None
