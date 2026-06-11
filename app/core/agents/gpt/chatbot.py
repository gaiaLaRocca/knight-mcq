import re
import nltk
import logging
import io
import sys
import warnings
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from logging.handlers import RotatingFileHandler
from time import time
# Import the specific warning type if possible, otherwise use Warning
try:
    from bs4 import GuessedAtParserWarning
except ImportError:
    GuessedAtParserWarning = Warning # Fallback if bs4 is not directly available/installed

from app.core.common.config import (
    GPT_NEO4J_URI as NEO4J_URI,
    GPT_NEO4J_USER as NEO4J_USER,
    GPT_NEO4J_PASSWORD as NEO4J_PASSWORD,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
    MAX_DEPTH as DEFAULT_MAX_DEPTH,
    DEFAULT_NO_DESCRIPTION,
    DEFAULT_ERROR_DESCRIPTION,
)
from app.core.common.neo4j_connection import Neo4jConnection
from app.core.agents.gpt.text_processing import extract_clean_special_terms, extract_triplets_from_response
from app.core.agents.gpt.term_description import query_term_description, generate_term_description, save_term_description
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# Import the new utility function
from app.core.utils.graph_utils import prune_non_wiki_descriptions

# Import QA Generation function
# from app.generation.qa_generation import generate_qa_from_graph # <-- Keep this commented at top level

# Reconfigure stdout to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Initialize Logging
# Use a named logger for better distinction
logger = logging.getLogger("gpt_agent")
logger.setLevel(logging.DEBUG) # Keep logger itself at DEBUG to allow DEBUG to file

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Include logger name in the format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console Handler (INFO level)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # <-- Revert to INFO
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler (DEBUG level)
log_file = os.path.join(log_dir, "chatbot.log")
file_handler = RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Suppress verbosity from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
# Suppress the specific parser warning from wikipedia/bs4
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# Thread-safe counter for how many descriptions were grounded on Wikipedia vs
# generated purely by the LLM (fallback). Processing is recursive + multithreaded,
# so all updates go through a lock. Reset at the start of each run.
class _WikiGroundingStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.wiki_grounded = 0
        self.llm_only = 0

    def reset(self):
        with self._lock:
            self.wiki_grounded = 0
            self.llm_only = 0

    def record(self, wiki_used: bool):
        with self._lock:
            if wiki_used:
                self.wiki_grounded += 1
            else:
                self.llm_only += 1

    def total(self) -> int:
        return self.wiki_grounded + self.llm_only

wiki_stats = _WikiGroundingStats()

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
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Global set to track which terms have already been processed (across all queries, if needed for other logic)
# processed_terms = set()  # Commenting out or remove if not used elsewhere for cross-query tracking
processed_descriptions = set()  # Track processed descriptions globally

# Wrapper class for LangChain LLM to match generate_qa_from_graph's expected interface
class SimpleLLMWrapper:
    def __init__(self, langchain_llm):
        self.langchain_llm = langchain_llm

    def generate(self, prompt: str) -> str | None:
        """Generate text using the underlying LangChain LLM"""
        try:
            response = self.langchain_llm.invoke([HumanMessage(content=prompt)]).content
            return response.strip() if response else None
        except Exception as e:
            logger.error(f"Error generating text via wrapper: {e}", exc_info=True)
            return None

def save_term_as_node(conn, term, description=None):
    """
    Save a term as a node in the Neo4j database with cleaned term names.
    Explicitly sets wiki_fact_checked to 'Yes' if saving the initial query/response term.
    """
    global processed_descriptions 
    term_clean = term.replace("\n", " ").strip().lower().replace("_", " ")

    if description:
        description_clean = description.replace("'", "\\'").replace("\n", " ").strip()
        if description_clean in processed_descriptions:
            logger.debug(f"Description for term '{term_clean}' already processed. Skipping save.")
            return
        # This is the initial query node being saved with the direct LLM response
        # Set fact_checked to Yes explicitly as requested
        query = """
        MERGE (t:Term {name: $term})
        SET t.description = $description, t.wiki_fact_checked = 'Yes'
        """
        parameters = {"term": term_clean, "description": description_clean}
    else:
        # If saving a term node without description (e.g., during triplet processing before description generation)
        # Don't set the flag yet, let save_term_description handle it.
        query = """
        MERGE (t:Term {name: $term})
        """
        parameters = {"term": term_clean}

    try:
        conn.execute_write(query, parameters=parameters)
        if description:
            processed_descriptions.add(description_clean)
            logger.info(f"Saved/updated term '{term_clean}' with description (Wiki Fact Checked: Yes - Initial Query).")
        else:
            logger.info(f"Saved/updated term '{term_clean}'.")
    except Exception as e:
        logger.error(f"Failed to save term '{term_clean}': {e}")

def create_relationship(conn, parent_term, child_term, relation="HAS_TERM"):
    """
    Create a relationship between terms in the Neo4j database based on the specified relation.
    """
    relation_clean = re.sub(r"\W|^(?=\d)", "_", relation).upper()
    parent_clean = parent_term.replace("_", " ").strip().lower()
    child_clean = child_term.replace("_", " ").strip().lower()
    query = f"""
    MATCH (p:Term {{name: $parent_term}})
    MATCH (c:Term {{name: $child_term}})
    MERGE (p)-[:{relation_clean}]->(c)
    RETURN p, c
    """
    try:
        logger.info(f"Attempting to create relationship '{relation_clean}' from '{parent_clean}' to '{child_clean}'.")
        result = conn.query(query, parameters={"parent_term": parent_clean, "child_term": child_clean})
        if result:
            logger.info(f"Successfully created relationship '{relation_clean}' from '{parent_clean}' to '{child_clean}'.") # Log success
        else:
            # This case might not happen often with MERGE if nodes exist, but good practice
            logger.warning(f"MERGE completed but maybe didn't create? Check relationship '{relation_clean}' from '{parent_clean}' to '{child_clean}'.")
    except Exception as e:
        logger.error(f"Failed to create relationship '{relation_clean}' from '{parent_clean}' to '{child_clean}': {e}")

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(Exception), reraise=True)
def process_triplet(conn, llm: ChatOpenAI, triplet, parent_term, depth, max_depth, current_query_processed_terms, original_response_text: str | None = None):
    """
    Process a triplet, passing the LLM instance and original response text for context.
    Also ensures terms are added to processed set before saving node to avoid log duplication.
    """
    global processed_descriptions 
    try:
        head = triplet.get("head", "").strip().lower()
        relation = triplet.get("relation", "").strip().lower()
        tail = triplet.get("tail", "").strip().lower()

        logger.debug(f"Processing triplet: Head='{head}', Relation='{relation}', Tail='{tail}' at depth {depth}")

        if not tail:
            logger.warning(f"Triplet with empty tail detected: {triplet}. Skipping.")
            return

        # Ensure nodes exist and have descriptions if needed, processing each term only once per query
        terms_to_process = {t for t in {head, tail} if t} # Get unique non-empty terms from triplet
        for term in terms_to_process:
            if term in current_query_processed_terms:
                logger.debug(f"Term '{term}' already processed in this query. Skipping node/description check.")
                continue 
            
            # Add to set *before* saving node
            current_query_processed_terms.add(term)
                
            logger.debug(f"Marked term '{term}' as processed for this query.")

            # Ensure the node exists 
            save_term_as_node(conn, term) 
            # logger.debug(f"Ensured node exists for term: '{term}'") # Log in save_term_as_node is sufficient

            # Check/Generate/Save Description
            if not query_term_description(conn, term):
                logger.debug(f"No description found for '{term}'. Generating...")
                parent_context_for_wiki = parent_term if term == tail else None

                # --- Determine source text for context (Revised for GPT agent) ---
                # Use original LLM response for depth 1, parent description (parent_term) otherwise
                source_for_context = original_response_text if depth == 1 else parent_term

                # Call generate_term_description, getting back description and flag
                # Pass the determined source text for context
                new_desc, wiki_used = generate_term_description(
                    _get_llm(), term, parent_term=parent_context_for_wiki, source_context_text=source_for_context
                )
                if new_desc and new_desc not in [DEFAULT_NO_DESCRIPTION, DEFAULT_ERROR_DESCRIPTION]:
                    # Track whether this description was grounded on Wikipedia or LLM-only
                    wiki_stats.record(wiki_used)
                    # Pass the flag to save_term_description
                    save_term_description(conn, term, new_desc, wiki_used)
                    # Logger info moved inside save_term_description
                    # logger.info(f"Generated and saved description for term '{term}'.") 
                else:
                    logger.warning(f"Failed to generate a valid description for term '{term}'.")
            else:
                 logger.debug(f"Description already exists for term '{term}'.")

            # Mark this term as processed for the current query (MOVED UP)
            # current_query_processed_terms.add(term)
            # logger.debug(f"Marked term '{term}' as processed for this query.")

        # Create Relationships (Nodes are guaranteed to exist by MERGE in save_term_as_node)
        final_relation = relation.upper() if relation else "HAS_TERM"
        # Relationship from parent (from previous level) to current tail
        if parent_term and tail:
             create_relationship(conn, parent_term, tail, final_relation) # Removed stats passing
        # Relationship within the triplet (head to tail)
        if head and tail and relation: # Only create if relation was specified
            create_relationship(conn, head, tail, final_relation) # Removed stats passing

        # Recursive processing for the tail term
        if depth >= max_depth:
            logger.info(f"Maximum recursion depth {max_depth} reached for term '{tail}'. Skipping sub-triplet extraction.")
        else:
            # Check if the tail term (which was processed above) is suitable for further exploration
            if tail in current_query_processed_terms: # Check if it was processed (it should have been)
                tail_description = query_term_description(conn, tail)
                if tail_description and tail_description not in [DEFAULT_NO_DESCRIPTION, DEFAULT_ERROR_DESCRIPTION]:
                    sub_triplets = extract_clean_special_terms(tail_description) # Assuming this function extracts triplets correctly
                    logger.debug(f"Extracted {len(sub_triplets)} sub-triplets from description of '{tail}'")
                    MAX_BRANCHES = 2
                    if len(sub_triplets) > MAX_BRANCHES:
                        logger.debug(f"Limiting sub-triplets for '{tail}' from {len(sub_triplets)} to {MAX_BRANCHES}.")
                        sub_triplets = sub_triplets[:MAX_BRANCHES]
                    
                    if sub_triplets:
                        # Use a ThreadPoolExecutor for concurrent processing of sub-triplets
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            # Pass original_response_text down in recursive calls
                            futures = {executor.submit(process_triplet, conn, _get_llm(), st, tail, depth + 1, max_depth, current_query_processed_terms, original_response_text): st for st in sub_triplets} # Removed stats passing
                            for future in as_completed(futures):
                                sub_triplet_info = futures[future]
                                try:
                                    future.result() # Wait for completion, handle exceptions
                                except Exception as e:
                                    logger.error(f"Error processing sub-triplet derived from '{tail}' (Triplet: {sub_triplet_info}): {e}", exc_info=True)
                    else:
                        logger.debug(f"No sub-triplets extracted from description of '{tail}'.")
                else:
                    logger.debug(f"No valid description available for '{tail}' to extract sub-triplets.")
            else:
                 logger.warning(f"Term '{tail}' was not marked as processed, skipping sub-triplet extraction. This might indicate an issue.")

    except Exception as e:
        # Log details including the triplet being processed when the error occurred
        triplet_info = f"Head='{triplet.get('head', 'N/A')}', Relation='{triplet.get('relation', 'N/A')}', Tail='{triplet.get('tail', 'N/A')}'"
        logger.error(f"Unexpected error processing triplet ({triplet_info}) at depth {depth}: {e}", exc_info=True)
        # Decide if we should raise or just log and continue
        # raise # Uncomment if errors should halt processing for the parent

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
        logger.error(f"Error generating initial LLM response: {e}")
        return DEFAULT_ERROR_DESCRIPTION

def extract_triplets_from_response(response):
    """
    Extract triplets from the LLM-generated response.
    """
    triplets = extract_clean_special_terms(response)
    logger.info(f"Extracted triplets: {triplets}")
    return triplets

def generate_response(user_input, conn, max_depth):
    """
    Generate a response and process the extracted triplets using the defined max_depth.
    Tracks processed terms within this specific query execution.
    Passes original response text for context handling.
    """
    try:
        # Reset Wikipedia-grounding stats for this run
        wiki_stats.reset()

        response = generate_llm_response(user_input)
        triplets = extract_triplets_from_response(response)

        # Track terms processed within this specific query
        current_query_processed_terms = set()
        
        # Ensure the initial user input term exists as a node
        user_input_term = user_input.lower().strip()
        save_term_as_node(conn, user_input_term, response) # Save with description from LLM
        current_query_processed_terms.add(user_input_term) # Mark initial term as processed
        logger.info(f"Saved initial prompt term '{user_input_term}' and added to processed set for this query.")

        # --- Timing Start ---
        start_time = time()
        graph_built = False
        # --- Timing Start ---
        
        if triplets:
            graph_built = True # Mark that we attempted building
            logger.info(f"Processing {len(triplets)} extracted triplets.")
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Pass the LLM instance, tracking set, original response text (Removed stats dict)
                futures = {executor.submit(process_triplet, conn, _get_llm(), t, user_input_term, 1, max_depth, current_query_processed_terms, response): t for t in triplets} # Removed stats passing
                for future in as_completed(futures):
                    triplet_info = futures[future]
                    try:
                        future.result() # Wait for completion and check for exceptions
                    except Exception as e:
                         # Log error including the specific triplet that failed
                        logger.error(f"Error processing top-level triplet {triplet_info}: {e}", exc_info=True)
            # --- Timing End & Log ---
            duration = time() - start_time
            logger.info(f"Triplet processing and graph building took {duration:.2f}s.")

            # --- Wikipedia grounding summary for this run ---
            total_desc = wiki_stats.total()
            if total_desc > 0:
                pct = 100.0 * wiki_stats.wiki_grounded / total_desc
                logger.info(
                    f"Wikipedia grounding summary: {wiki_stats.wiki_grounded}/{total_desc} "
                    f"descriptions grounded on Wikipedia ({pct:.1f}%), "
                    f"{wiki_stats.llm_only} generated by LLM only."
                )
            else:
                logger.info("Wikipedia grounding summary: no new descriptions were generated this run.")
        else:
            logger.debug("No triplets extracted from the response.")
            # --- Log Zero Time if No Triplets ---
            if not graph_built: # Only log if we didn't even start building
                 duration = time() - start_time # Should be near zero
                 logger.info(f"No triplets to process, graph building time: {duration:.2f}s.")
                 # Removed stats log for zero triplets case
            # --- Log Zero Time if No Triplets ---
            
        return response
    except Exception as e:
        logger.error(f"Unexpected error in generate_response for input '{user_input}': {e}", exc_info=True)
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
    """Main chat loop for interaction."""
    # Use the imported default depth, allow modification per iteration
    current_max_depth = DEFAULT_MAX_DEPTH 
    
    print("Chatbot: Hi! How can I help you?")
    if topic:
        print(f"Chatbot: Current session topic: '{topic}'")
    else:
        print("Chatbot: No specific topic set for this session.")
    
    # Initial help message can be printed here or inside the loop
    # print("Chatbot: Type 'help' for available commands.") 
    logger.info("Chatbot started.")

    while True:
        # --- Ask for Max Depth BEFORE the main prompt --- 
        print("\nChatbot: Set the exploration depth for the next question/action.")
        print(f"         (Controls how many steps to explore in the graph for answers. Default: {current_max_depth})")
        depth_input = input(f"Chatbot: Enter max depth (or press Enter to use {current_max_depth}): ").strip()
        if depth_input:
            try:
                new_depth = int(depth_input)
                if new_depth >= 0:
                    current_max_depth = new_depth
                    logger.info(f"Max depth for next action set to {current_max_depth}.")
                else:
                    print(f"Chatbot: Invalid input. Using previous depth: {current_max_depth}.")
                    # Keep current_max_depth as is
            except ValueError:
                print(f"Chatbot: Invalid input. Using previous depth: {current_max_depth}.")
                # Keep current_max_depth as is
        else:
             logger.info(f"Using default/previous max depth: {current_max_depth}.")
        # Let the user know the active depth
        print(f"Chatbot: Exploration depth for the next action is {current_max_depth}.")

        # --- Print Help/Commands --- 
        print("Chatbot: Available commands: help, /generate_qa, show related to [term], bye/exit, /prune_descriptions")
        print("         Or just ask a question!")
        
        # --- Get User Command/Question --- 
        user_input = input("You: ").strip()

        # --- Command Handling --- 
        if user_input.lower() in ["bye", "exit"]:
            print("Chatbot: Goodbye!")
            logger.info("Chatbot session ended by user.")
            break
        elif user_input.lower() == "help":
            # Updated help message
            print("Chatbot: Available commands:")
            print(f" - To ask a question: [Your question] (Current exploration depth: {current_max_depth})")
            print(" - To show related terms: 'show related to [term]'")
            print(" - To generate QA pairs interactively: '/generate_qa'") 
            print(" - To prune non-wiki descriptions: '/prune_descriptions'")
            print(" - To exit: 'bye' or 'exit'")
            print("Note: Exploration depth is set before each command prompt.")
        elif user_input.lower().startswith("show related to"):
            term = user_input[len("show related to "):].strip().lower()
            if term:
                related = get_related_terms(conn, term)
                if related:
                    related_strs = [f"{r['relation']} -> {r['related_term']}" for r in related]
                    print(f"Chatbot: Terms related to '{term}': {related_strs}")
                else:
                    print(f"Chatbot: No terms found related to '{term}' or term not found.")
            else:
                print("Chatbot: Please specify a term. Use 'show related to [term]'.")

        elif user_input.lower() == "/prune_descriptions":
            logger.info("User initiated description pruning for wiki_fact_checked='No' nodes.")
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
                    logger.error(f"Error calling prune_non_wiki_descriptions: {e}", exc_info=True)
                    print(f"Chatbot: An unexpected error occurred calling the pruning function: {e}")
            else:
                print("Chatbot: Pruning cancelled.")
                logger.info("User cancelled description pruning.")
        
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
            except ImportError as ie:
                logger.critical(f"Failed to import qa_generation module: {ie}")
                print("Chatbot: Error: Could not load the QA generation module.")
            except Exception as e:
                logger.error(f"Error during QA generation process: {e}", exc_info=True)
                print(f"Chatbot: An error occurred during QA generation: {e}")
            # --- End of QA Generation Flow --- 

        else: # Treat as a standard question for the graph/LLM
            # Use the max_depth set at the start of this loop iteration
            response = generate_response(user_input, conn, current_max_depth) 
            print(f"Chatbot: {response}")

    logger.info("Exiting chat loop.")

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
