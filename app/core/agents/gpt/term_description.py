import logging
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
        # Define the standard System Prompt (8-point structure)
        base_prompt = """You are a subject-matter expert in a scientific field. Your task is to provide detailed, thorough, and academically structured explanations about terms provided by the user. Each term should be explained exhaustively using the following structure:

1.  Definition and Scope – Provide a precise, scientific definition of the term. Outline its general scope, including the boundaries and extent of its meaning and use.
2.  Domains of Use – Identify all relevant scientific, technical, or professional domains where this term plays a key role. Specify the fields in which this concept is critical and explain its importance in each.
3.  Subfields and Disciplines – Break the term down into its major subfields, branches, or areas of study. Provide a brief but comprehensive overview of each subfield, including key principles, practices, and contributors.
4.  Key Concepts and Mechanisms – Describe the most important ideas, mechanisms, or processes associated with this term in various contexts. Explain how these ideas interconnect.
5.  Real-World Applications – Discuss the major practical applications of this concept in different spheres, such as industry, healthcare, environmental science, etc.
6.  Case Studies and Examples – Provide specific case studies, examples, or practical demonstrations of the term in action. Show how it is applied in real-world scenarios.
7.  Related and Overlapping Terms – Identify related or similar terms and concepts. Clarify how they are connected, and explain any subtle distinctions.
8.  Current Research and Trends – Briefly cover the current research directions, innovations, and debates around this concept. Mention any ongoing advancements or challenges in the field.

Your explanation should be clear, well-organized, scientifically accurate, and educational. Assume that the user is unfamiliar with the term, so explain each concept thoroughly. Use precise language and cite notable research, when possible. Dive deeply into subtopics as needed to provide a full understanding of the term's scope and implications.
""" # <<< Ensure this closing triple quote is indented correctly

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
            task_instruction_wiki = f"Now, please apply the structured explanation approach defined in the system prompt to explain the term: '{term}'."
            context_instruction_wiki = f"""Use the following Wikipedia context as the primary source for your explanation, structuring your response according to the system prompt guidelines:
--- Wikipedia Context ---
{wikipedia_summary}
--- End Wikipedia Context ---"""
            parent_hint_wiki = f"Also consider its relationship to the parent term '{parent_term}'." if parent_term else ""
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
            # Base Persona and Structure Prompt (Moved outside if/else)
            # base_prompt = """ ... """

            # Task Specific Instruction (for no-wiki case)
            task_instruction = f"\n\nNow, please apply this structure to explain the term: '{term}'."

            # Add context if available (parent term or source text)
            context_hint = None
            if parent_term:
                context_hint = f"Consider its relationship to the parent term '{parent_term}'."
            if source_context_text:
                context_hint = (context_hint + "\n" if context_hint else "") + f"Additional context from source text: {source_context_text}"

            if context_hint:
                task_instruction += f"\n{context_hint}"

            # Combine Base Prompt (System) and Task Instruction (Human)
            system_prompt_content = base_prompt # Use the common system prompt
            human_prompt_content = task_instruction
            
            # Invoke with separate System and Human messages
            logger.debug(f"Generating description for '{term}' using structured prompt (System + Human). System Prompt: {system_prompt_content[:200]}... Human Prompt: {human_prompt_content[:300]}...")
            response = llm.invoke([
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=human_prompt_content)
            ]).content
            # *** END: Modified Fallback Prompt Handling ***
        
        description = response.strip()
        if description:
            logger.info(f"Generated description for term '{term}' (Wikipedia context: {'Yes' if wikipedia_context_used else 'No'})")
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
