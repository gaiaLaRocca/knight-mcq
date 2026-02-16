import re
import nltk
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.common.config import (
    REBEL_NEO4J_URI as NEO4J_URI,
    REBEL_NEO4J_USER as NEO4J_USER,
    REBEL_NEO4J_PASSWORD as NEO4J_PASSWORD,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
    MAX_DEPTH,
    DEFAULT_NO_DESCRIPTION,
    DEFAULT_ERROR_DESCRIPTION,
)
from app.core.common.neo4j_connection import Neo4jConnection
from app.core.agents.rebel.text_processing import extract_clean_special_terms
from app.core.agents.rebel.term_description import (
    query_term_description,
    generate_term_description,
    save_term_description,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys
import io
import os
import warnings
from bs4 import GuessedAtParserWarning
import json # Add back for parsing validation response
from app.core.utils.graph_utils import prune_non_wiki_descriptions 

# Reconfigure stdout to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Suppress the specific parser warning from wikipedia/bs4
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# Initialize Logging
# Use a named logger for better distinction
logger = logging.getLogger("rebel_agent") 
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Include logger name in the format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
log_file = os.path.join(log_dir, "chatbot.log")
file_handler = RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
# No need to set formatter again here if it was set on the root logger before
# If file_handler needs its own format (it uses the logger's formatter by default)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Suppress verbosity from external libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# LLM lazy-initialized so tests can import without OPENAI_* env (see _get_llm below)
_llm_cache = None


def _get_llm():
    """Lazy-initialize LLM so tests can import this module without OPENAI_* env set."""
    global _llm_cache
    if _llm_cache is None:
        if not OPENAI_MODEL:
            raise ValueError("OPENAI_MODEL must be set in environment (e.g. in .env)")
        kwargs = dict(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.4)
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        _llm_cache = ChatOpenAI(**kwargs)
    return _llm_cache

# NLP tools
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)

def save_term_as_node(conn, term, description=None):
    """
    Save a term as a node in the Neo4j database with cleaned term names.
    Optionally sets the description and wiki_fact_checked flag (to No for initial query).
    """
    term_clean = re.sub(r"\s+", " ", term.replace("\n", " ")).strip().lower()

    if description:
        # This is the initial query node being saved with the direct LLM response
        # Set fact_checked to No explicitly
        query = """
        MERGE (t:Term {name: $term})
        SET t.description = $description, t.wiki_fact_checked = 'No'
        """
        description_clean = description.replace("'", "\\'").replace("\n", " ").strip()
        parameters = {"term": term_clean, "description": description_clean}
    else:
        # If saving without description initially, don't set the flag
        query = """
        MERGE (t:Term {name: $term})
        """
        parameters = {"term": term_clean}

    try:
        conn.execute_write(query, parameters=parameters)
        if description:
            logger.info(f"REBEL: Saved term '{term_clean}' as node with description (Wiki Fact Checked: No - Initial Query).")
        else:
            logger.info(f"REBEL: Saved term '{term_clean}' as node without description.")
    except Exception as e:
        logger.error(f"REBEL: Failed to save term '{term_clean}': {e}")

def create_relationship(conn, parent_term, child_term, relation="HAS_TERM"):
    """
    Create a relationship between terms in the Neo4j database based on the specified relation.
    """
    relation_clean = re.sub(r"\W|^(?=\d)", "_", relation).upper()
    parent_clean = parent_term.strip().lower()
    child_clean = child_term.strip().lower()
    query = f"""
    MATCH (p:Term {{name: $parent_term}})
    MATCH (c:Term {{name: $child_term}})
    MERGE (p)-[:{relation_clean}]->(c)
    RETURN p, c
    """
    try:
        result = conn.query(
            query, parameters={"parent_term": parent_clean, "child_term": child_clean}
        )
        if result:
            logger.info(
                f"Created relationship '{relation_clean}' from '{parent_clean}' to '{child_clean}'."
            )
        else:
            logger.warning(
                f"Failed to create relationship '{relation_clean}' from '{parent_clean}' to '{child_clean}' because one or both nodes do not exist."
            )
    except Exception as e:
        logger.error(
            f"Failed to create relationship '{relation_clean}' from '{parent_clean}' to '{child_clean}': {e}"
        )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def process_triplet(conn, llm: ChatOpenAI, triplet, parent_term, depth, max_depth, current_query_processed_terms, original_response_text: str | None = None):
    """
    Process a triplet, passing the LLM instance and original response text for context.
    Uses per-query tracking set.
    """
    try:
        head = triplet.get("head", "").strip().lower()
        relation = triplet.get("relation", "").strip().lower()
        tail = triplet.get("tail", "").strip().lower()

        logger.debug(f"REBEL: Processing triplet: Head='{head}', Relation='{relation}', Tail='{tail}' at depth {depth}")

        if not tail:
            logger.warning(f"REBEL: Triplet with empty tail detected: {triplet}. Skipping.")
            return

        # Process terms only once per query
        terms_to_process = {t for t in {head, tail} if t}
        for term in terms_to_process:
            if term in current_query_processed_terms:
                logger.debug(f"REBEL: Term '{term}' already processed in this query. Skipping.")
                continue
                
            # Mark as processed *before* saving node to prevent concurrent logs
            current_query_processed_terms.add(term) 
            logger.debug(f"REBEL: Marked term '{term}' as processed for this query.")

            # Ensure node exists (without description first)
            save_term_as_node(conn, term) 
            # logger.debug(f"REBEL: Ensured node exists for term: '{term}'") # Log in save_term_as_node is sufficient

            # Check if term needs description generation with RAG
            if not query_term_description(conn, term):
                logger.debug(f"REBEL: No description found for '{term}'. Generating...")
                parent_context_for_wiki = parent_term if term == tail else None
                
                # --- Determine source text for context (Revised) ---
                # Use original LLM response for depth 1, parent description (parent_term) otherwise
                source_for_context = original_response_text if depth == 1 else parent_term

                # Call generate_term_description with LLM, getting back description and flag
                new_desc, wiki_used = generate_term_description(
                    _get_llm(), term, parent_term=parent_context_for_wiki, source_context_text=source_for_context
                )
                if new_desc and new_desc not in [DEFAULT_NO_DESCRIPTION, DEFAULT_ERROR_DESCRIPTION]:
                    # Pass the flag to save_term_description
                    save_term_description(conn, term, new_desc, wiki_used)
                    # No need for extra log here, save_term_description logs success
                else:
                    logger.warning(f"REBEL: Failed to generate a valid description for term '{term}'.")
            else:
                 logger.debug(f"REBEL: Description already exists for term '{term}'.")

        # Create relationships
        final_relation = relation.upper() if relation else "HAS_TERM"
        # Ensure parent_term exists before creating relationship
        if parent_term and tail:
             # We assume parent_term node was created in a previous step or earlier in the loop
             create_relationship(conn, parent_term, tail, final_relation)
        # Ensure head and tail exist before creating relationship
        if head and tail and relation:
            # Nodes for head and tail were ensured earlier in the loop
            create_relationship(conn, head, tail, final_relation)

        # Control recursive sub-triplet extraction based on max_depth
        if depth >= max_depth:
            logger.info(f"REBEL: Maximum recursion depth {max_depth} reached for term '{tail}'. Skipping sub-triplet extraction.")
        else:
            if tail in current_query_processed_terms:
                tail_description = query_term_description(conn, tail)
                if tail_description and tail_description not in [DEFAULT_NO_DESCRIPTION, DEFAULT_ERROR_DESCRIPTION]:
                    # Use raw sub-triplets directly (no validation step)
                    sub_triplets = extract_clean_special_terms(tail_description)
                    logger.debug(f"REBEL: Extracted {len(sub_triplets)} sub-triplets from '{tail}'")
                    
                    MAX_BRANCHES = 3
                    # Limit the raw sub-triplets
                    if len(sub_triplets) > MAX_BRANCHES:
                        logger.debug(f"REBEL: Limiting sub-triplets for '{tail}' from {len(sub_triplets)} to {MAX_BRANCHES}.")
                        sub_triplets = sub_triplets[:MAX_BRANCHES]

                    # Process the raw (but limited) sub-triplets
                    if sub_triplets:
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            # Pass original_response_text down in recursive calls
                            futures = {executor.submit(process_triplet, conn, _get_llm(), st, tail, depth+1, max_depth, current_query_processed_terms, original_response_text): st for st in sub_triplets} 
                            for future in as_completed(futures):
                                sub_triplet_info = futures[future]
                                try:
                                    future.result()
                                except Exception as e:
                                    logger.error(f"REBEL: Error processing sub-triplet derived from '{tail}' (Triplet: {sub_triplet_info}): {e}", exc_info=True)
                    else:
                        logger.debug(f"REBEL: No sub-triplets extracted from description of '{tail}'.")
                else:
                    logger.debug(f"REBEL: No valid description available for '{tail}', skipping sub-triplet extraction.")
            else:
                logger.warning(f"REBEL: Term '{tail}' was not marked as processed? Skipping sub-triplet extraction.")

    except Exception as e:
        triplet_info = f"Head='{triplet.get('head', 'N/A')}', Relation='{triplet.get('relation', 'N/A')}', Tail='{triplet.get('tail', 'N/A')}'"
        logger.error(f"REBEL: Unexpected error processing triplet ({triplet_info}) at depth {depth}: {e}", exc_info=True)
        raise # Re-raise to allow retry decorator to work

def generate_llm_response(user_input):
    """
    Generate a response from the LLM using the user's input.
    """
    prompt = re.sub(r"\s+", " ", user_input.replace("\n", " ")).strip()
    try:
        response = _get_llm().invoke([HumanMessage(content=prompt)]).content
        logger.info(f"Generated initial LLM response: {response[:200]}...")
        return response
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return DEFAULT_ERROR_DESCRIPTION

def extract_triplets_from_response(response):
    """
    Extract triplets from the LLM-generated response.
    """
    triplets = extract_clean_special_terms(response)
    logger.info(f"Extracted triplets: {triplets}")
    return triplets

# --- Re-add Triplet Validation Function ---
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
def validate_triplet_with_llm(llm: ChatOpenAI, triplet: dict, source_text: str) -> bool:
    """Uses LLM to validate if a triplet is directly supported by the source text."""
    head = triplet.get('head', '')
    relation = triplet.get('relation', '')
    tail = triplet.get('tail', '')
    if not all([head, relation, tail]):
        return False # Invalid triplet structure

    prompt = f"""
    Source Text:
    {source_text}
    
    Triplet:
    Head: {head}
    Relation: {relation}
    Tail: {tail}
    
    Question: Is the relationship '{head} {relation} {tail}' directly and accurately stated in the Source Text? 
    (Do not infer or assume relationships not explicitly mentioned).
    Answer ONLY with 'Yes' or 'No'.
    """
    try:
        # Use a lower temperature for validation for more deterministic answers
        validation_llm = llm.with_config({"temperature": 0.1})
        response = validation_llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        logger.debug(f"REBEL: Triplet Validation for ({head},{relation},{tail}): LLM response '{response}'")
        is_valid = response.startswith('yes')
        if not is_valid:
             logger.info(f"REBEL: Triplet ({head},{relation},{tail}) failed validation against source.")
        return is_valid
    except Exception as e:
        logger.error(f"REBEL: Error during LLM triplet validation for ({head},{relation},{tail}): {e}")
        return False # Default to invalid on error
# --- End Re-add Function ---

def generate_response(user_input, conn, max_depth):
    """
    Generate REBEL response and process triplets, passing LLM and tracking terms.
    Includes robust triplet validation and passes original response for context.
    """
    try:
        response = generate_llm_response(user_input)
        # Extract raw triplets
        raw_triplets = extract_triplets_from_response(response)

        # --- Re-add Validation Step for Initial Triplets ---
        validated_triplets = []
        if raw_triplets:
             logger.info(f"REBEL: Validating {len(raw_triplets)} initial triplets...")
             with ThreadPoolExecutor(max_workers=5) as validator_executor:
                  future_to_triplet = {validator_executor.submit(validate_triplet_with_llm, _get_llm(), t, response): t for t in raw_triplets}
                  for future in as_completed(future_to_triplet):
                       triplet = future_to_triplet[future]
                       try:
                            if future.result():
                                 validated_triplets.append(triplet)
                       except Exception as exc:
                            logger.error(f"REBEL: Initial triplet validation generated an exception for {triplet}: {exc}")
             logger.info(f"REBEL: Kept {len(validated_triplets)} initial triplets after validation.")
        # --- End Re-add Validation Step ---

        # Use per-query tracking set
        current_query_processed_terms = set()
        
        user_input_term = user_input.lower().strip()
        # Save initial node with wiki_fact_checked='No'
        save_term_as_node(conn, user_input_term, response)
        current_query_processed_terms.add(user_input_term)
        logger.info(f"REBEL: Saved initial prompt term '{user_input_term}'.")

        # Use validated triplets 
        if validated_triplets: 
            logger.info(f"REBEL: Processing {len(validated_triplets)} validated extracted triplets.")
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Pass LLM, tracking set, and the original 'response' text to initial process_triplet calls
                futures = {executor.submit(process_triplet, conn, _get_llm(), t, user_input_term, 1, max_depth, current_query_processed_terms, response): t for t in validated_triplets} 
                for future in as_completed(futures):
                    triplet_info = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                         logger.error(f"REBEL: Error processing top-level triplet {triplet_info}: {e}", exc_info=True)
        else:
            logger.debug("REBEL: No triplets extracted from the response.")
            
        # The REBEL agent typically doesn't return the LLM response directly to user?
        # Depending on desired behavior, you might return nothing, or status, or the extracted graph info.
        # For consistency in testing, let's return the original response for now.
        return response 
    except Exception as e:
        logger.error(f"REBEL: Unexpected error in generate_response for input '{user_input}': {e}", exc_info=True)
        return DEFAULT_ERROR_DESCRIPTION

def get_related_terms(conn, term):
    """
    Retrieve related terms and their relationship types from the Neo4j database.
    """
    query = """
    MATCH (p:Term {name: $term})-[r]->(c:Term)
    RETURN type(r) AS relation, c.name AS related_term
    """
    try:
        results = conn.query(query, parameters={"term": term})
        related_terms = [{"relation": r["relation"], "related_term": r["related_term"]} for r in results]
        logger.info(f"Retrieved related terms for '{term}': {related_terms}")
        return related_terms
    except Exception as e:
        logger.error(f"Error retrieving related terms for '{term}': {e}")
        return []

# --- Input Helper Functions ---

def get_int_input(prompt: str, min_value: int = 1, allow_none: bool = False) -> int | None:
    """Prompts user for an integer input with validation."""
    while True:
        user_input = input(prompt).strip()
        if allow_none and not user_input:
            return None
        try:
            value = int(user_input)
            if value >= min_value:
                return value
            else:
                print(f"Chatbot: Please enter an integer greater than or equal to {min_value}.")
        except ValueError:
            print("Chatbot: Invalid input. Please enter a number.")

def get_yes_no_input(prompt: str, default_yes: bool) -> bool:
    """Prompts user for a Yes/No input."""
    default_indicator = "(Y/n)" if default_yes else "(y/N)"
    prompt_with_default = f"{prompt} {default_indicator}: "
    while True:
        user_input = input(prompt_with_default).strip().lower()
        if not user_input: # User pressed Enter
            return default_yes
        if user_input in ['y', 'yes']:
            return True
        if user_input in ['n', 'no']:
            return False
        print("Chatbot: Please answer with 'yes' or 'no' (or press Enter for default).")

def get_float_input(prompt: str, min_val: float, max_val: float, default: float) -> float:
    """Prompts user for a float input within a range."""
    prompt_with_default = f"{prompt} (default: {default}): "
    while True:
        user_input = input(prompt_with_default).strip()
        if not user_input:
            return default
        try:
            value = float(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Chatbot: Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Chatbot: Invalid input. Please enter a number.")

# --- End Input Helper Functions ---

def chat(conn, topic):
    """
    Main chatbot loop for user interaction.
    """
    print("Chatbot: Hi! How can I help you?")
    logger.info("Chatbot started.")

    current_max_depth = MAX_DEPTH  # Initialize with default MAX_DEPTH

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["bye", "exit"]:
                print("Chatbot: Bye!")
                logger.info("Chatbot session ended by user.")
                break
            elif user_input.lower().startswith("show related to "):
                term = user_input[16:].strip().lower()
                related_terms = get_related_terms(conn, term)
                if related_terms:
                    print(f"Chatbot: Terms related to '{term}':")
                    for rel in related_terms:
                        print(f" - {rel['relation']} -> {rel['related_term']}")
                else:
                    print(f"Chatbot: No related terms found for '{term}'.")
            elif user_input.lower().startswith("set max depth "):
                try:
                    new_depth = int(user_input[14:].strip())
                    if new_depth < 0:
                        raise ValueError("Depth cannot be negative.")
                    current_max_depth = new_depth
                    logger.info(f"MAX_DEPTH updated to {new_depth} by user.")
                    print(f"Chatbot: MAX_DEPTH set to {new_depth}.")
                except ValueError as ve:
                    print(f"Chatbot: Invalid depth value. {ve}")
                except Exception as e:
                    logger.error(f"Error setting MAX_DEPTH: {e}")
                    print("Chatbot: Failed to set MAX_DEPTH due to an error.")
            elif user_input.lower() == "/prune_descriptions":
                logger.info("REBEL: User initiated description pruning for wiki_fact_checked='No' nodes.")
                print("\nChatbot: Attempting to prune descriptions for non-Wikipedia-sourced nodes...")
                confirmation = get_yes_no_input("Chatbot: This will set descriptions to null for nodes where wiki_fact_checked='No'. This cannot be undone easily. Proceed?", default_yes=False)
                if confirmation:
                    try:
                        # Call the utility function
                        updated_count = prune_non_wiki_descriptions(conn)
                        
                        # Handle the return value
                        if updated_count is not None and updated_count > 0:
                            print(f"Chatbot: Successfully pruned descriptions for {updated_count} node(s).")
                        elif updated_count == 0:
                            print("Chatbot: No descriptions needed pruning (or no relevant nodes found).")
                        else: # updated_count is None (error)
                            print("Chatbot: An error occurred during the pruning process. Check logs.")

                    except Exception as e:
                        # Catch potential errors if the function call itself fails unexpectedly
                        logger.error(f"REBEL: Error calling prune_non_wiki_descriptions: {e}", exc_info=True)
                        print(f"Chatbot: An unexpected error occurred calling the pruning function: {e}")
                else:
                    print("Chatbot: Pruning cancelled.")
                    logger.info("REBEL: User cancelled description pruning.")
            elif user_input.lower() == "/generate_qa": 
                # --- Interactive QA Generation Flow --- 
                # This flow remains the same, using its own complexity settings internally
                logger.info("User initiated interactive QA generation.")
                print("\nChatbot: Okay, let's configure the QA generation.")
                
                # Confirm Topic Usage
                if topic:
                    print(f"Chatbot: We'll generate pairs relevant to the current session topic: '{topic}'")
                else:
                    print("Chatbot: No specific topic is set for this session.")

                # 1. Complexity
                print("\nChatbot: First, specify the path complexity.")
                print(" - 'exact': Generate questions ONLY from paths with a specific number of relationships.")
                print(" - 'max': Generate questions from paths with UP TO a specific number of relationships (e.g., max=3 finds paths of length 1, 2, and 3).")
                exact_complexity = None
                max_complexity_val = None
                while True:
                    mode = input("Chatbot: Use 'exact' or 'max' complexity? (default: max): ").strip().lower()
                    if not mode or mode == 'max':
                        max_complexity_val = get_int_input("Chatbot: Enter the maximum number of relationships (path length): ", min_value=1)
                        if max_complexity_val:
                            break
                    elif mode == 'exact':
                        exact_complexity = get_int_input("Chatbot: Enter the exact number of relationships (path length): ", min_value=1)
                        if exact_complexity:
                            max_complexity_val = exact_complexity 
                            break
                    else:
                        print("Chatbot: Please type 'exact' or 'max'.")

                # 2. Limit
                print("\nChatbot: Optionally, limit the total number of graph paths processed (useful for large graphs or testing).")
                limit = get_int_input("Chatbot: Enter max paths to process (or press Enter for no limit): ", min_value=1, allow_none=True)

                # 3. Validation
                print("\nChatbot: QA pairs can be validated by an LLM for grammar, answerability, and topic relevance.")
                skip_validation = get_yes_no_input("Chatbot: Skip validation?", default_yes=False)
                validation_sample_rate = 1.0
                if not skip_validation:
                    print("Chatbot: You can validate all pairs (rate=1.0) or a random sample (e.g., rate=0.5 for 50%).")
                    validation_sample_rate = get_float_input("Chatbot: Enter validation sample rate (0.0 to 1.0): ", min_val=0.0, max_val=1.0, default=1.0)

                # 4. Reverse QA
                print("\nChatbot: By default, questions often have the end node of the path as the answer.")
                generate_reverse = get_yes_no_input("Chatbot: Also attempt to generate reverse questions (start node as answer)?", default_yes=False)

                # --- Call Generation --- 
                print("\nChatbot: Configuration complete. Starting generation...")
                logger.info(f"Starting QA generation with collected parameters: mode={'exact' if exact_complexity else 'max'}, complexity={exact_complexity or max_complexity_val}, limit={limit}, skip_validation={skip_validation}, rate={validation_sample_rate}, reverse={generate_reverse}, topic={topic}")
                print("Chatbot: This might take some time...")
                try:
                    from app.generation.qa_generation import generate_qa_from_graph
                    logger.info("Dynamically imported generate_qa_from_graph.")
                    
                    effective_max_complexity = max_complexity_val 
                    effective_exact_complexity = exact_complexity

                    complexity_mode_str = f"exact complexity={effective_exact_complexity}" if effective_exact_complexity is not None else f"max complexity={effective_max_complexity}"
                    limit_str = f"limit={limit}" if limit else "no limit"
                    validation_str = f"validation skipped" if skip_validation else f"validation rate={validation_sample_rate}"
                    reverse_str = "Reverse QA: Enabled" if generate_reverse else "Reverse QA: Disabled"
                    topic_str = f"Topic: '{topic}'" if topic else "No Topic"
                    print(f"Chatbot: Calling generate_qa_from_graph ({complexity_mode_str}, {limit_str}, {validation_str}, {reverse_str}, {topic_str})...")

                    generated_pairs = generate_qa_from_graph(
                        neo4j_conn=conn, 
                        llm_client=_get_llm(),
                        max_complexity=effective_max_complexity, 
                        exact_complexity=effective_exact_complexity, 
                        limit=limit,
                        skip_validation=skip_validation,
                        validation_sample_rate=validation_sample_rate,
                        topic=topic,
                        generate_reverse=generate_reverse
                    )
                    logger.info(f"QA generation process finished. Generated {len(generated_pairs)} pairs.")
                    # The function generate_qa_from_graph already prints success/failure/save messages
                except ImportError as ie:
                    logger.critical(f"Failed to import qa_generation module: {ie}")
                    print("Chatbot: Error: Could not load the QA generation module.")
                except Exception as e:
                    logger.error(f"Error during QA generation process: {e}", exc_info=True)
                    print(f"Chatbot: An error occurred during QA generation: {e}")
                # --- End of QA Generation Flow --- 
            elif user_input.lower().startswith("help") or user_input.lower().startswith("commands"):
                print("Chatbot: Available commands:")
                print(" - To ask a question: [Your question]")
                print(" - To show related terms: 'show related to [term]'")
                print(" - To set maximum depth: 'set max depth [number]'")
                print(" - To generate QA pairs interactively: '/generate_qa'")
                print(" - To prune non-wiki descriptions: '/prune_descriptions'")
                print(" - To exit: 'bye' or 'exit'")
            else:
                response = generate_response(user_input, conn, current_max_depth)
                print(f"Chatbot: {response}")
        except KeyboardInterrupt:
            print("\nChatbot: Bye!")
            logger.info("Chatbot session terminated by KeyboardInterrupt.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in chat loop: {e}")
            print("Chatbot: I'm sorry, something went wrong. Please try again.")

if __name__ == "__main__":
    # Prompt for topic *before* connecting/starting chat
    session_topic = input("Please enter the main topic for this session (or press Enter for none): ").strip()
    if not session_topic:
        session_topic = None # Ensure it's None if empty
        logger.info("No specific topic provided for this session.")
    else:
        logger.info(f"Session topic set to: '{session_topic}'")

    try:
        with Neo4jConnection(uri=NEO4J_URI, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as conn:
            # Ensure index on Term.name for efficient querying
            try:
                index_query = "CREATE INDEX IF NOT EXISTS FOR (t:Term) ON (t.name)"
                conn.execute_write(index_query)
                logger.info("Ensured index on Term.name.")
            except Exception as e:
                logger.error(f"Error creating index on Term.name: {e}")

            try:
                # Pass the collected topic to the chat function
                chat(conn, session_topic)
            except Exception as e:
                logger.error(f"Error during chat session: {e}") # Corrected log message
    except Exception as e:
        logger.critical(f"Failed to establish Neo4j connection: {e}")