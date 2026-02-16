# app/generation/qa_generation.py

import logging
import json
import time # Import time module
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import random
import uuid # <-- Import uuid
import re # <-- Import re
from langchain_core.messages import SystemMessage, HumanMessage # <-- Ensure imports

logger = logging.getLogger(__name__)

# --- System Prompts for MCQ Generation ---
SYSTEM_PROMPT_QA_FORWARD = """You are a structured question generation system. Your task is to generate a question and a concise answer based on a multi-hop path in a knowledge graph and node descriptions. 
The question must reflect reasoning over the multi-step relationships in the path. 
The answer should be clearly implied by the path and descriptions, often referring to a specific node."""

SYSTEM_PROMPT_QA_REVERSE = """You are a reasoning assistant generating reverse questions from knowledge graph paths. 
Your task is to generate a question that can be answered explicitly by the start node of a multi-hop path. 
Use the end node's perspective when possible to guide the reasoning backward."""

# --- System Prompt for MCQ Validation ---
SYSTEM_PROMPT_MCQ_VALIDATION = '''You are a rigorous MCQ-validation assistant.

TASK
Evaluate a four-option multiple-choice question (MCQ) using **only** the information supplied in the "Source Information" block.
Answer with six YES/NO (or N/A) tags—one per line—in the exact order and casing shown below.

CHECKLIST
1. GRAMMAR_FLUENCY        Is the Question spelled and phrased correctly and clearly?
2. SINGLE_CORRECT_KEY      Is exactly one option marked as correct?
3. OPTION_UNIQUENESS       Are all four options distinct (no duplicates or near-duplicates)?
4. ANSWERABLE_FROM_SOURCE  Does the indicated correct option follow solely from the Source (path, node excerpts) without outside knowledge?
5. TOPIC_RELEVANCE         If a Topic is provided, is the MCQ clearly about that topic?
6. ETHICS_PRIVACY_SAFE     Does the MCQ respect ethical standards and privacy? (No hateful, disallowed, or personal-data content.)

STRICT OUTPUT FORMAT
Grammar_Fluency: [YES/NO]  
Single_Correct_Key: [YES/NO]  
Option_Uniqueness: [YES/NO]  
Answerable_From_Source: [YES/NO]  
Topic_Relevant: [YES/NO or N/A]  
Ethics_Privacy_Safe: [YES/NO]  

Return **nothing else**—no explanations.

––––––––––––––––––––––––––––––––––––––––
FEW-SHOT EXAMPLES  (for internal guidance only)
––––––––––––––––––––––––––––––––––––––––

► EXAMPLE 1 – All criteria satisfied
User Input
Question: In photosynthesis, which component supplies the energy that powers chloroplast reactions?  
Option A: Oxygen  
Option B: Sunlight  
Option C: Carbon dioxide  
Option D: Chlorophyll  
Correct Answer Key: B  
Topic (optional): Photosynthesis  
Source Information  
  Path: sunlight → powers → photosynthesis  
  Start Node 'sunlight' (excerpt): "…electromagnetic radiation from the Sun…"  
  End   Node 'photosynthesis' (excerpt): "…process by which green plants convert light energy…"  

Expected Assistant Output  
Grammar_Fluency: YES  
Single_Correct_Key: YES  
Option_Uniqueness: YES  
Answerable_From_Source: YES  
Topic_Relevant: YES  
Ethics_Privacy_Safe: YES  

────────────────────────────────────────

► EXAMPLE 2 – Duplicate option (fails OPTION_UNIQUENESS)
User Input
Question: Which gas is released during photosynthesis?  
Option A: Oxygen  
Option B: Oxygen  
Option C: Carbon dioxide  
Option D: Nitrogen  
Correct Answer Key: A  
Topic (optional): Photosynthesis  
Source Information  
  Path: photosynthesis → produces → oxygen  
  Start Node 'photosynthesis' (excerpt): "…process converts CO₂ and water into glucose and releases O₂…"  
  End   Node 'oxygen' (excerpt): "…a diatomic gas essential for respiration…"


Expected Assistant Output  
Grammar_Fluency: YES  
Single_Correct_Key: YES  
Option_Uniqueness: NO  
Answerable_From_Source: YES  
Topic_Relevant: YES  
Ethics_Privacy_Safe: YES  

────────────────────────────────────────

► EXAMPLE 3 – Multiple correct answers (fails SINGLE_CORRECT_KEY & ANSWERABLE_FROM_SOURCE)
User Input
Question: Which of the following are noble gases?  
Option A: Helium  
Option B: Neon  
Option C: Argon  
Option D: Oxygen  
Correct Answer Key: A  
Topic (optional): Noble gases  
Source Information  
  Path: noble_gases → include → helium, neon, argon  
  Start Node 'noble_gases' (excerpt): "…group 18 elements known for inertness…"  
  End   Node 'argon' (excerpt): "…another noble gas element…"  

Expected Assistant Output  
Grammar_Fluency: YES  
Single_Correct_Key: NO  
Option_Uniqueness: YES  
Answerable_From_Source: NO  
Topic_Relevant: YES  
'''

def generate_qa_from_graph(neo4j_conn, llm_client, max_complexity=2, exact_complexity=None, limit=None, skip_validation=False, validation_sample_rate=1.0, topic: str | None = None, generate_reverse: bool = False):
    """
    Main function to orchestrate QA pair generation from graph paths.
    Uses ThreadPoolExecutor for concurrency within generate_qa_from_paths.
    
    Parameters:
    - neo4j_conn: Neo4j connection object
    - llm_client: LLM client object
    - max_complexity: Maximum path length to fetch if exact_complexity is None.
    - exact_complexity: If set, only generate QA for paths of this exact length.
    - limit: Maximum total number of paths to process (for testing/limiting cost).
    - skip_validation: If True, skip the validation step entirely.
    - validation_sample_rate: Between 0.0-1.0, percentage of items to validate.
    - topic: Optional string topic to keep QA pairs relevant to.
    - generate_reverse: If True, attempt to generate reverse QA pairs as well.
    """
    print("[Debug] Entered generate_qa_from_graph function.") 
    
    reverse_msg = " (Reverse QA Enabled)" if generate_reverse else ""
    topic_msg = f" (Topic: '{topic}')" if topic else ""
    complexity_mode = f"exact complexity={exact_complexity}" if exact_complexity is not None else f"max complexity={max_complexity}"
    limit_msg = f" with total path limit={limit}" if limit else ""
    validation_msg = " (validation skipped)" if skip_validation else f" (validation rate={validation_sample_rate})"
    # Updated print message
    print(f"Starting QA generation from paths ({complexity_mode}){limit_msg}{validation_msg}{topic_msg}{reverse_msg}...") 
    logger.info(f"Starting QA generation orchestration (Paths) ({complexity_mode}){limit_msg}{validation_msg}{topic_msg}{reverse_msg}...")

    all_qa_pairs = [] # Initialize as list

    # --- Generate QA from Paths --- 
    try:
        logger.info(f"Calling generate_qa_from_paths ({complexity_mode}, limit={limit}, topic='{topic}', reverse={generate_reverse})...")
        start_time = time.time()
        # Call the new path generation function, passing topic and generate_reverse
        all_qa_pairs = generate_qa_from_paths(
            neo4j_conn, 
            llm_client, 
            max_complexity=max_complexity, 
            exact_complexity=exact_complexity,
            limit=limit,
            topic=topic, # Pass topic
            generate_reverse=generate_reverse # Pass reverse flag
        )
        duration = time.time() - start_time
        logger.info(f"generate_qa_from_paths returned {len(all_qa_pairs)} pairs in {duration:.2f}s.")
        print(f"Generated {len(all_qa_pairs)} pairs from paths in {duration:.2f}s.")
    except Exception as e:
        logger.error(f"Error during generate_qa_from_paths call: {e}", exc_info=True)
        print(f"Error generating QA from paths: {e}")
        all_qa_pairs = [] # Ensure it's a list even on error

    # --- Removed Triple Generation Call --- 

    # --- Validation Section (operates on all_qa_pairs from paths) --- 
    logger.info(f"Total generated pairs before validation: {len(all_qa_pairs)}")
    print(f"Total pairs before validation: {len(all_qa_pairs)}")
    
    validated_qa_pairs = all_qa_pairs 
    if not skip_validation and all_qa_pairs: 
        logger.info(f"Starting validation with sample rate {validation_sample_rate} and topic '{topic}'...")
        print(f"Starting validation (sample rate={validation_sample_rate})...")
        start_time = time.time()
        # Pass topic to validation
        validated_qa_pairs = validate_qa_pairs(all_qa_pairs, llm_client, sample_rate=validation_sample_rate, topic=topic)
        duration = time.time() - start_time
        logger.info(f"Validation completed in {duration:.2f}s.")
        print(f"Validation completed in {duration:.2f}s.")
    elif skip_validation:
        logger.info("Validation skipped as requested.")
        print("Validation skipped as requested.")
    elif not all_qa_pairs: 
        logger.info("No QA pairs generated from paths, skipping validation.")
        print("No QA pairs generated, skipping validation.")

    # --- Save results --- 
    if validated_qa_pairs:
        # Add topic to the filename? Or maybe as a field within the json?
        # Let's keep the filename simple for now.
        filename = "generated_qa_pairs.json"
        logger.info(f"Saving {len(validated_qa_pairs)} validated pairs to {filename}...")
        save_qa_pairs(validated_qa_pairs, filepath=filename) 
    else:
        logger.warning("No validated QA pairs generated to save.")
        print("No validated QA pairs to save.")

    # Updated final message
    logger.info(f"Finished QA generation orchestration (Paths). Generated {len(validated_qa_pairs)} validated QA pairs ({complexity_mode}, Limit={limit}{reverse_msg}).")
    print(f"Finished QA generation (Paths). Saved {len(validated_qa_pairs)} pairs ({complexity_mode}, Limit={limit}{reverse_msg}).")
    return validated_qa_pairs

# Add a timeout decorator for LLM calls
def timeout_decorator(timeout_seconds=30):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import concurrent.futures
            import threading
            
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                error_msg = f"Function call timed out after {timeout_seconds} seconds"
                logger.error(error_msg)
                raise TimeoutError(error_msg)
            
            if exception[0]:
                raise exception[0]
                
            return result[0]
        return wrapper
    return decorator

# Wrap the original generate method with timeout
def safe_generate(llm_client, messages: list, timeout_seconds=30, max_retries=2):
    """Safely generate text with timeout and retries using a list of messages."""
    
    @timeout_decorator(timeout_seconds)
    def _generate_with_timeout():
        # Assumes llm_client has an invoke method that accepts a list of messages
        return llm_client.invoke(messages).content
    
    for attempt in range(max_retries + 1):
        try:
            return _generate_with_timeout()
        except TimeoutError:
            if attempt < max_retries:
                logger.warning(f"LLM call timed out (attempt {attempt+1}/{max_retries+1}), retrying...")
                time.sleep(1)  # Brief pause before retry
            else:
                logger.error(f"LLM call failed after {max_retries+1} attempts")
                return None
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None

MIN_QUESTION_LEN = 10
MAX_QUESTION_LEN = 200
MIN_ANSWER_LEN = 1

def _format_combined_validation_prompt(mcq_data: dict, source_type: str, source_details: dict, topic: str | None = None) -> str:
    """ Creates the human prompt for the new 6-point MCQ validation task. """
    question_text = mcq_data.get('question', '[Missing Question]')
    options = mcq_data.get('options', {})
    option_a = options.get('A', '')
    option_b = options.get('B', '')
    option_c = options.get('C', '')
    option_d = options.get('D', '')
    correct_answer_key = mcq_data.get('correct_answer_key', '[Missing Key]')
    topic_or_blank = topic if topic else ""
    
    # Extract path and node details (similar to before)
    nodes_list = source_details.get('nodes', [])
    relationships = source_details.get('relationships', [])
    path_representation = "[Invalid Path Data]"
    start_node_name = "[N/A]"
    start_node_desc = "[N/A]"
    end_node_name = "[N/A]"
    end_node_desc = "[N/A]"

    if isinstance(nodes_list, list) and nodes_list and isinstance(nodes_list[0], dict):
        start_node_info = nodes_list[0]
        end_node_info = nodes_list[-1]
        start_node_name = start_node_info.get('name', '?')
        start_node_desc = start_node_info.get('description') or "No description available."
        end_node_name = end_node_info.get('name', '?')
        end_node_desc = end_node_info.get('description') or "No description available."
        
        if relationships and len(nodes_list) == len(relationships) + 1:
            path_str_parts = [f"({start_node_name})"]
            for i, rel_type in enumerate(relationships):
                next_node_name = nodes_list[i+1].get('name', '?') if i+1 < len(nodes_list) else "?"
                path_str_parts.append(f"-[:{rel_type}]->({next_node_name})")
            path_representation = "".join(path_str_parts)
        else:
            path_representation = "[Length Mismatch or Missing Relationships]"
            
    # Construct the new human prompt
    prompt = f"""Evaluate the following MCQ based ONLY on the Source Information.

Question: {question_text}
Option A: {option_a}
Option B: {option_b}
Option C: {option_c}
Option D: {option_d}
Correct Answer Key: {correct_answer_key}
Topic (optional): {topic_or_blank}

Source Information
  Path: {path_representation}
  Start Node '{start_node_name}' (excerpt): {start_node_desc[:150]}…
  End   Node '{end_node_name}'   (excerpt): {end_node_desc[:150]}…

Provide your evaluation in the required six-line format.
"""
    return prompt.strip() 

def parse_combined_validation_response(response_text):
    """ Parses the new 6-line LLM response for MCQ validation.
        Returns tuple: (grammar_ok, single_key_ok, uniqueness_ok, answerable_ok, topic_ok, ethics_ok)
    """
    # Defaults
    grammar_ok = False
    single_key_ok = False
    uniqueness_ok = False
    answerable_ok = False
    topic_ok = False # Assume relevant unless explicitly N/A or NO
    ethics_ok = False

    if not response_text:
        logger.warning("Received empty response text for validation parsing.")
        return grammar_ok, single_key_ok, uniqueness_ok, answerable_ok, topic_ok, ethics_ok

    try:
        lines = response_text.strip().split('\n')
        result_map = {}
        
        # Expected keys (lowercase for case-insensitive matching)
        expected_keys = [
            'grammar_fluency', 
            'single_correct_key', 
            'option_uniqueness', 
            'answerable_from_source', 
            'topic_relevant', 
            'ethics_privacy_safe'
        ]

        for line in lines:
            line_lower = line.lower().strip()
            if ":" in line_lower:
                # Split only on the first colon
                key, value = line.split(":", 1)
                # Clean key: lowercase, replace underscores/spaces
                clean_key = key.strip().lower().replace(' ', '_')
                # Clean value: uppercase, strip whitespace
                clean_value = value.strip().upper()
                result_map[clean_key] = clean_value

        # Check each expected key
        grammar_ok = result_map.get('grammar_fluency') == 'YES'
        single_key_ok = result_map.get('single_correct_key') == 'YES'
        uniqueness_ok = result_map.get('option_uniqueness') == 'YES'
        answerable_ok = result_map.get('answerable_from_source') == 'YES'
        ethics_ok = result_map.get('ethics_privacy_safe') == 'YES'
        
        # Topic relevance: YES or N/A are considered OK. Only explicit NO fails.
        topic_verdict = result_map.get('topic_relevant')
        topic_ok = topic_verdict != 'NO' # True if YES, N/A, or key missing
        
        # Log the parsed results for debugging
        logger.debug(f"Parsed validation: Grammar={grammar_ok}, SingleKey={single_key_ok}, Unique={uniqueness_ok}, Answerable={answerable_ok}, Topic={topic_ok} (Verdict: {topic_verdict}), Ethics={ethics_ok}")

        return grammar_ok, single_key_ok, uniqueness_ok, answerable_ok, topic_ok, ethics_ok

    except Exception as e:
        logger.warning(f"Could not parse new 6-line validation response: '{response_text}'. Error: {e}")
        # Return default False values on parsing error
        return False, False, False, False, False, False 

def validate_qa_pairs(qa_pairs, llm_client, sample_rate=1.0, topic: str | None = None):
    """
    Validates generated QA pairs for quality and correctness using ThreadPoolExecutor.
    Uses combined LLM check for grammar, answerability, and topic relevance.
    
    Parameters:
    - qa_pairs: List of QA pairs to validate
    - llm_client: LLM client for validation
    - sample_rate: Float between 0.0-1.0, percentage of pairs to validate (1.0 = all)
    - topic: Optional string topic to check relevance against.
    """
    total_pairs = len(qa_pairs)
    topic_msg = f" against topic '{topic}'" if topic else ""
    logger.info(f"Starting validation for {total_pairs} generated QA pairs (sample rate: {sample_rate}){topic_msg}...")
    print(f"Validating {total_pairs} QA pairs with sample rate {sample_rate}{topic_msg}...")
    
    # --- Sampling --- 
    pairs_to_validate_indices = list(range(total_pairs))
    if sample_rate < 1.0:
        sample_size = max(1, int(total_pairs * sample_rate))
        if sample_size < total_pairs:
            pairs_to_validate_indices = random.sample(pairs_to_validate_indices, sample_size)
            logger.info(f"Sampling {sample_size} out of {total_pairs} pairs for validation")
            print(f"Sampling {sample_size} pairs for validation...")
            
    pairs_to_validate_map = {idx: qa_pairs[idx] for idx in pairs_to_validate_indices}
    num_to_validate = len(pairs_to_validate_map)
    validated_pairs_map = {} # Store validated pairs by original index
    rejected_count = 0
    processed_count = 0 # For LLM validation progress
    progress_interval = max(1, min(100, num_to_validate // 10)) # Show progress every ~10%

    if not llm_client:
        logger.error("LLM client is required for validation but was not provided. Skipping LLM checks.")
        # If skipping, consider all *sampled* pairs as *potentially* valid structurally
        # Let's perform only structural checks and return those that pass
        structurally_valid_pairs = []
        structurally_rejected_count = 0
        for idx, pair in pairs_to_validate_map.items():
             question = pair.get("question")
             options = pair.get("options")
             correct_key = pair.get("correct_answer_key")
             is_structurally_valid = True
             rejection_reason = []
             if not question or not isinstance(question, str) or len(question) < MIN_QUESTION_LEN or len(question) > MAX_QUESTION_LEN:
                 is_structurally_valid = False; rejection_reason.append("Invalid/missing/length question")
             if not options or not isinstance(options, dict) or len(options) < 2 or not all(isinstance(v, str) and v for v in options.values()):
                 is_structurally_valid = False; rejection_reason.append("Invalid/missing/empty options")
             if not correct_key or not isinstance(correct_key, str) or correct_key not in options:
                 is_structurally_valid = False; rejection_reason.append(f"Invalid/missing correct_answer_key ('{correct_key}')")
             
             if is_structurally_valid:
                 structurally_valid_pairs.append(pair)
             else:
                 structurally_rejected_count += 1
                 logger.debug(f"Rejected pair #{idx} on structural check (LLM skipped): Reason(s): {'; '.join(rejection_reason)}.")
        
        # Add back non-sampled pairs if any
        if sample_rate < 1.0:
             non_sampled_pairs = [qa_pairs[i] for i in range(total_pairs) if i not in pairs_to_validate_map]
             structurally_valid_pairs.extend(non_sampled_pairs)
             logger.info(f"Added {len(non_sampled_pairs)} non-sampled pairs back (structural check only). Final count: {len(structurally_valid_pairs)}")
        
        logger.info(f"LLM validation skipped. Returning {len(structurally_valid_pairs)} structurally valid pairs ({structurally_rejected_count} rejected structurally).")
        return structurally_valid_pairs

    # --- Concurrent Validation ---    
    MAX_WORKERS_VALIDATION = 15 # Set concurrent validation workers (tune based on RPM limits)
    logger.info(f"Using ThreadPoolExecutor with max_workers={MAX_WORKERS_VALIDATION} for validation.")

    futures = {}
    pairs_submitted_to_llm = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_VALIDATION) as executor:
        # First pass: Structural checks and submit valid ones to executor
        for idx, pair in pairs_to_validate_map.items():
            question = pair.get("question")
            options = pair.get("options")
            correct_key = pair.get("correct_answer_key")
            source_details = pair.get("source_details")
            
            is_structurally_valid = True
            structural_rejection_reason = []

            # Perform structural checks
            if not question or not isinstance(question, str) or len(question) < MIN_QUESTION_LEN or len(question) > MAX_QUESTION_LEN:
                is_structurally_valid = False; structural_rejection_reason.append("Invalid/missing/length question")
            if not options or not isinstance(options, dict) or len(options) < 2 or not all(isinstance(v, str) and v for v in options.values()):
                is_structurally_valid = False; structural_rejection_reason.append("Invalid/missing/empty options")
            if not correct_key or not isinstance(correct_key, str) or correct_key not in options:
                is_structurally_valid = False; structural_rejection_reason.append(f"Invalid/missing correct_answer_key ('{correct_key}')")
            if not source_details: # Need source details for LLM check
                is_structurally_valid = False; structural_rejection_reason.append("Missing source details for LLM validation")

            if is_structurally_valid:
                # Only submit structurally valid pairs for LLM validation
                pairs_submitted_to_llm += 1
                source_type = pair.get("source_type")
                validation_human_prompt = _format_combined_validation_prompt(pair, source_type, source_details, topic=topic) 
                validation_messages = [
                    SystemMessage(content=SYSTEM_PROMPT_MCQ_VALIDATION),
                    HumanMessage(content=validation_human_prompt)
                ]
                future = executor.submit(safe_generate, llm_client, validation_messages, timeout_seconds=45)
                futures[future] = idx # Map future back to original index
            else:
                # Log rejection based on structural checks immediately
                rejected_count += 1
                logger.debug(f"Rejected pair #{idx} pre-LLM check: Reason(s): {'; '.join(structural_rejection_reason)}. Q: {str(question)[:50]}... Key: {correct_key}")
        
        logger.info(f"Submitted {pairs_submitted_to_llm} pairs (out of {num_to_validate} sampled) for LLM validation.")

        # Process completed futures
        for future in as_completed(futures):
            original_idx = futures[future]
            pair = pairs_to_validate_map[original_idx] # Get the original pair data
            is_valid = True # Assume valid initially for LLM check part
            llm_rejection_reason = []
            
            try:
                validation_response_text = future.result() # Get result from future
                
                if validation_response_text:
                    logger.debug(f"Raw validation response for pair #{original_idx}:\n---\n{validation_response_text}\n---")
                    grammar_ok, single_key_ok, uniqueness_ok, answerable_ok, topic_ok, ethics_ok = parse_combined_validation_response(validation_response_text)
                    
                    # Apply all checks from the parser
                    if not grammar_ok: is_valid = False; llm_rejection_reason.append("LLM grammar/clarity check failed")
                    if not single_key_ok: is_valid = False; llm_rejection_reason.append("LLM single key check failed")
                    if not uniqueness_ok: is_valid = False; llm_rejection_reason.append(f"LLM uniqueness check failed")
                    if not answerable_ok: is_valid = False; llm_rejection_reason.append(f"LLM answerability check failed")
                    if not topic_ok: is_valid = False; llm_rejection_reason.append(f"LLM topic relevance check failed (Topic: '{topic}')")
                    if not ethics_ok: is_valid = False; llm_rejection_reason.append(f"LLM ethics check failed")
                else:
                    logger.warning(f"MCQ Validation LLM call timed out or failed for pair #{original_idx}, accepting it anyway")
                    is_valid = True # Keep it valid if LLM call failed/timed out

            except Exception as exc:
                logger.error(f"Error processing validation future for pair #{original_idx}: {exc}", exc_info=True)
                is_valid = False # Reject if future processing itself fails
                llm_rejection_reason.append(f"Future processing error: {exc}")

            # --- Decision based on LLM checks ---            
            processed_count += 1
            if is_valid:
                validated_pairs_map[original_idx] = pair # Store accepted pair by index
            else:
                rejected_count += 1
                logger.debug(f"Rejected pair #{original_idx} post-LLM check: Reason(s): {'; '.join(llm_rejection_reason)}. Q: {pair.get('question', '')[:50]}... Key: {pair.get('correct_answer_key')}")

            # Log progress
            if processed_count % progress_interval == 0 or processed_count == pairs_submitted_to_llm:
                progress_pct = processed_count / pairs_submitted_to_llm * 100 if pairs_submitted_to_llm > 0 else 100
                logger.info(f"LLM Validation progress: {processed_count}/{pairs_submitted_to_llm} pairs completed ({progress_pct:.1f}%). Current rejected: {rejected_count}")
                print(f"LLM Validation progress: {processed_count}/{pairs_submitted_to_llm} pairs completed ({progress_pct:.1f}%). Current rejected: {rejected_count}")

    # --- Combine results --- 
    final_validated_pairs = []
    # Add LLM-validated pairs in original order
    for idx in sorted(validated_pairs_map.keys()):
        final_validated_pairs.append(validated_pairs_map[idx])
        
    # Add the non-sampled pairs back if necessary
    if sample_rate < 1.0:
        non_sampled_pairs = [qa_pairs[i] for i in range(total_pairs) if i not in pairs_to_validate_map]
        final_validated_pairs.extend(non_sampled_pairs)
        logger.info(f"Added {len(non_sampled_pairs)} non-sampled pairs to the final result")
    
    # Calculate acceptance rate based on pairs actually submitted to LLM
    num_submitted_to_llm = pairs_submitted_to_llm
    num_accepted_by_llm = len(validated_pairs_map)
    acceptance_rate = (num_accepted_by_llm / num_submitted_to_llm * 100) if num_submitted_to_llm > 0 else 0 # Avoid division by zero
    total_structurally_rejected = num_to_validate - num_submitted_to_llm
    logger.info(f"Validation complete. Accepted {len(final_validated_pairs)} out of {total_pairs} total pairs.")
    logger.info(f"(Structurally rejected pre-LLM: {total_structurally_rejected}. Submitted to LLM: {num_submitted_to_llm}. Accepted post-LLM: {num_accepted_by_llm} -> {acceptance_rate:.1f}% acceptance rate of submitted)")
    print(f"Validation complete. Accepted {len(final_validated_pairs)} out of {total_pairs} QA pairs")
    
    return final_validated_pairs

def save_qa_pairs(qa_pairs, filepath="generated_qa_pairs.json"):
    """
    Saves the validated QA pairs to a JSON file.
    """
    logger.info(f"Attempting to save {len(qa_pairs)} QA pairs to {filepath}...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved QA pairs to {filepath}")
    except IOError as e:
        logger.error(f"IOError saving QA pairs to {filepath}: {e}")
    except TypeError as e:
        logger.error(f"TypeError during JSON serialization: {e}. Check data types in qa_pairs.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving QA pairs: {e}") 

# --- NEW MULTI-HOP PROMPT FUNCTION ---
def _format_multihop_qa_prompt(path_data, topic: str | None = None):
    """Formats the Human prompt for the LLM to generate multiple-choice QA pairs.
    Args:
        path_data (dict): A dictionary representing the path...
        topic (str | None): The session topic, if any.
    Returns:
        str: The formatted Human prompt string, or None if path data is invalid.
    """
    nodes = path_data.get("nodes")
    relationships = path_data.get("relationships")
    if not nodes or not relationships or len(nodes) != len(relationships) + 1:
        logger.error(f"Invalid path data received for multi-hop prompt: {path_data}")
        return None
    start_node = nodes[0]
    end_node = nodes[-1]
    path_str_parts = [f"({start_node.get('name', 'Unknown')})"]
    for i, rel_type in enumerate(relationships):
        next_node_name = nodes[i+1].get('name', 'Unknown')
        path_str_parts.append(f"-[:{rel_type}]->({next_node_name})")
    path_representation = "".join(path_str_parts)
    start_desc = start_node.get('description') or "No description available."
    end_desc = end_node.get('description') or "No description available."
    topic_instruction = f"IMPORTANT: The generated Question and Options MUST be relevant to the overall topic: '{topic}'." if topic else ""

    # Construct the Human Prompt content
    human_prompt = f"""Follow the instructions in the system prompt to generate a multiple-choice question based on the provided path and node descriptions.

Example 1:
Path: (Paris)-[:CAPITAL_OF]->(France)-[:MEMBER_OF]->(European Union)
Start Node: Paris | Description: The capital city of France.
End Node: European Union | Description: A political and economic union.

Question: The country whose capital is Paris is a member of which union?
A) NATO
B) European Union
C) United Nations
D) African Union
Correct Answer: B

Example 2:
Path: (Hafiz)-[:EXPRESS_THEMES_OF]->(Love)-[:EXAMPLE_OF]->(Emotion)
Start Node: Hafiz | Description: A 14th-century Persian poet.
End Node: Emotion | Description: A complex state of feeling.

Question: What kind of concept is exemplified by a theme expressed in Hafiz's poetry?
A) Historical Event
B) Emotion
C) Scientific Theory
D) Geographical Location
Correct Answer: B

{topic_instruction}

Now, generate for the following:
Path: {path_representation}
Start Node: {start_node.get('name')} | Description: {start_desc}
End Node: {end_node.get('name')} | Description: {end_desc}

IMPORTANT: You MUST generate exactly four options (A, B, C, D) and indicate the single correct answer key. Adhere strictly to the output format below.

Output:
Question: [Your generated question reflecting the multi-step path]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A, B, C, or D]
"""
    return human_prompt.strip()

# --- NEW REVERSE MULTI-HOP PROMPT FUNCTION ---
def _format_multihop_qa_prompt_reverse(path_data, topic: str | None = None):
    """Formats the Human prompt for the LLM to generate REVERSE multiple-choice QA pairs.
    Args:
        path_data (dict): Path dictionary...
        topic (str | None): The session topic...
    Returns:
        str: The formatted Human prompt string, or None if path data is invalid.
    """
    nodes = path_data.get("nodes")
    relationships = path_data.get("relationships")
    if not nodes or not relationships or len(nodes) != len(relationships) + 1:
        logger.error(f"Invalid path data received for reverse multi-hop prompt: {path_data}")
        return None
    start_node = nodes[0]
    end_node = nodes[-1]
    start_node_name = start_node.get('name', 'Unknown')
    path_str_parts = [f"({start_node_name})"]
    for i, rel_type in enumerate(relationships):
        next_node_name = nodes[i+1].get('name', 'Unknown')
        path_str_parts.append(f"-[:{rel_type}]->({next_node_name})")
    path_representation = "".join(path_str_parts)
    start_desc = start_node.get('description') or "No description available."
    end_desc = end_node.get('description') or "No description available."
    topic_instruction = f"IMPORTANT: The generated Question and Options MUST be relevant to the overall topic: '{topic}'." if topic else ""

    # Construct the Human Prompt content for reverse multiple-choice
    human_prompt_reverse = f"""Follow the instructions in the system prompt to generate a multiple-choice question where the START NODE ('{start_node_name}') is the correct answer.

Example 1:
Path: (Paris)-[:CAPITAL_OF]->(France)-[:MEMBER_OF]->(European Union)
Start Node: Paris | Description: The capital city of France.
End Node: European Union | Description: A political and economic union.

Question: Which capital city belongs to a country that is a member of the European Union?
A) Berlin
B) Rome
C) Paris
D) Madrid
Correct Answer: C

Example 2:
Path: (Hafiz)-[:EXPRESS_THEMES_OF]->(Love)-[:EXAMPLE_OF]->(Emotion)
Start Node: Hafiz | Description: A 14th-century Persian poet.
End Node: Emotion | Description: A complex state of feeling.

Question: Emotion is exemplified by a theme expressed in the poetry of which Persian poet?
A) Hafiz
B) Rumi
C) Saadi
D) Omar Khayyam
Correct Answer: A

{topic_instruction}

Now, generate for the following:
Path: {path_representation}
Start Node: {start_node_name} | Description: {start_desc}
End Node: {end_node.get('name')} | Description: {end_desc}

IMPORTANT: You MUST generate exactly four options (A, B, C, D) and indicate the single correct answer key (which MUST correspond to the option containing the Start Node name '{start_node_name}'). Adhere strictly to the output format below.

Output:
Question: [Generated question targeting the start node]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [Letter corresponding to the option containing the exact text '{start_node_name}']
"""
    return human_prompt_reverse.strip()

# --- NEW MCQ PARSER FUNCTION ---
def _parse_mcq_output(text: str | None) -> dict | None:
    """Parses LLM output for Question, Options (A-D), and Correct Answer Key."""
    if not text:
        return None

    try:
        question_match = re.search(r"Question:\s*(.*)", text, re.IGNORECASE)
        option_a_match = re.search(r"\nA\)\s*(.*)", text, re.IGNORECASE)
        option_b_match = re.search(r"\nB\)\s*(.*)", text, re.IGNORECASE)
        option_c_match = re.search(r"\nC\)\s*(.*)", text, re.IGNORECASE)
        option_d_match = re.search(r"\nD\)\s*(.*)", text, re.IGNORECASE)
        correct_answer_match = re.search(r"\nCorrect Answer:\s*([A-D])", text, re.IGNORECASE | re.MULTILINE)

        if not all([question_match, option_a_match, option_b_match, option_c_match, option_d_match, correct_answer_match]):
            logger.warning(f"Could not parse all MCQ components from response: {text[:200]}...")
            return None

        question = question_match.group(1).strip()
        options = {
            "A": option_a_match.group(1).strip(),
            "B": option_b_match.group(1).strip(),
            "C": option_c_match.group(1).strip(),
            "D": option_d_match.group(1).strip(),
        }
        correct_key = correct_answer_match.group(1).strip().upper()
        
        # Basic validation
        if not question or not all(options.values()) or correct_key not in options:
             logger.warning(f"Parsed MCQ components are invalid: Q={question}, Opts={options}, Key={correct_key}")
             return None

        return {
            "question": question,
            "options": options,
            "correct_answer_key": correct_key
        }
    except Exception as e:
        logger.error(f"Error parsing MCQ response: {e}. Response: {text[:200]}...", exc_info=True)
        return None

# --- UPDATED MULTI-HOP PATH PROCESSING FUNCTION ---
def _process_single_path(path_data, llm_client, topic: str | None = None, generate_reverse: bool = False):
    """Processes a single path to generate forward and optionally reverse MULTIPLE-CHOICE QA pairs."""
    generated_pairs = []
    # Imports are ensured at module level now
    # from langchain_core.messages import SystemMessage, HumanMessage 

    try:
        nodes = path_data.get('nodes', [])
        relationships = path_data.get('relationships', [])
        complexity = len(relationships)
        start_node_name = nodes[0].get('name', 'Unknown') if nodes else 'Unknown'

        if complexity == 0 or not nodes or len(nodes) != complexity + 1:
            logger.warning(f"Received invalid path data: {path_data}")
            return []

        # Prepare shared source details
        source_details = {
            'nodes': nodes,
            'relationships': relationships,
            'fact_checked': {
                'start_node': nodes[0].get('fact_checked', 'Unknown') if nodes else 'Unknown',
                'end_node': nodes[-1].get('fact_checked', 'Unknown') if nodes else 'Unknown'
            }
        }

        # --- Generate Forward MCQ Pair (Always) ---
        human_prompt_forward = _format_multihop_qa_prompt(path_data, topic=topic)
        if human_prompt_forward:
            messages_forward = [
                SystemMessage(content=SYSTEM_PROMPT_QA_FORWARD),
                HumanMessage(content=human_prompt_forward)
            ]
            start_time = time.time()
            # Call safe_generate with the list of messages and increased timeout
            generated_qa_output = safe_generate(llm_client, messages_forward, timeout_seconds=60)
            duration = time.time() - start_time
            logger.debug(f"LLM call (Forward MCQ, complexity {complexity}) for path starting at '{start_node_name}' took {duration:.2f}s")

            parsed_mcq = _parse_mcq_output(generated_qa_output)
            if parsed_mcq:
                pair_id = str(uuid.uuid4())
                generated_pairs.append({
                    "id": pair_id,
                    "question": parsed_mcq["question"],
                    "options": parsed_mcq["options"],
                    "correct_answer_key": parsed_mcq["correct_answer_key"],
                    "source_type": "multi_hop_path_mcq", # New source type
                    "complexity": complexity,
                    "source_details": source_details
                })
            else:
                logger.warning(f"Failed to parse forward MCQ output for path '{start_node_name}'.")
        # else: Do nothing if forward human prompt formatting failed

        # --- Generate Reverse MCQ Pair (Optional) ---
        if generate_reverse:
            human_prompt_reverse = _format_multihop_qa_prompt_reverse(path_data, topic=topic)
            if human_prompt_reverse:
                messages_reverse = [
                    SystemMessage(content=SYSTEM_PROMPT_QA_REVERSE),
                    HumanMessage(content=human_prompt_reverse)
                ]
                start_time = time.time()
                # Call safe_generate with the list of messages and increased timeout
                generated_qa_output_rev = safe_generate(llm_client, messages_reverse, timeout_seconds=60)
                duration = time.time() - start_time
                logger.debug(f"LLM call (Reverse MCQ, complexity {complexity}) for path starting at '{start_node_name}' took {duration:.2f}s")

                parsed_mcq_rev = _parse_mcq_output(generated_qa_output_rev)
                if parsed_mcq_rev:
                    # Sanity check: Does the indicated correct answer match the start node?
                    correct_option_text = parsed_mcq_rev["options"].get(parsed_mcq_rev["correct_answer_key"])
                    if correct_option_text and correct_option_text.lower() == start_node_name.lower():
                         pair_id = str(uuid.uuid4())
                         generated_pairs.append({
                            "id": pair_id,
                            "question": parsed_mcq_rev["question"],
                            "options": parsed_mcq_rev["options"],
                            "correct_answer_key": parsed_mcq_rev["correct_answer_key"],
                            "source_type": "multi_hop_path_mcq_reverse", # New source type
                            "complexity": complexity,
                            "source_details": source_details
                         })
                    else:
                         logger.warning(f"Reverse MCQ parser indicated key '{parsed_mcq_rev['correct_answer_key']}' ('{correct_option_text}') which does not match start node '{start_node_name}'. Discarding.")
                else:
                    logger.warning(f"Failed to parse reverse MCQ output for path '{start_node_name}'.")
            # else: Do nothing if reverse human prompt formatting failed

        return generated_pairs # Return list (might contain 0, 1, or 2 pairs)

    except Exception as e:
        logger.error(f"Error processing multi-hop path for MCQ: {e}", exc_info=True)
        return []

def generate_qa_from_paths(neo4j_conn, llm_client, max_complexity=2, exact_complexity=None, limit=None, max_workers=10, topic: str | None = None, generate_reverse: bool = False):
    """
    Generates QA pairs based on graph paths.
    If exact_complexity is set, fetches and processes only paths of that length.
    Otherwise, fetches and processes paths up to max_complexity.
    Processes up to 'limit' paths if specified.
    Passes topic and generate_reverse flag to individual path processors.
    """
    # Determine log message based on mode
    reverse_msg = ", Reverse QA Enabled" if generate_reverse else ""
    topic_msg = f", Topic: '{topic}'" if topic else ""
    complexity_mode = f"exact complexity={exact_complexity}" if exact_complexity is not None else f"max complexity={max_complexity}"
    logger.info(f"Starting CONCURRENT QA generation from paths ({complexity_mode}{topic_msg}{reverse_msg})...")
    all_generated_pairs = []
    paths_data = []

    try:
        t_start_paths = time.time()
        # Pass exact_complexity directly to find_paths
        logger.info(f"--> Attempting to fetch paths ({complexity_mode})...")
        paths_data = neo4j_conn.find_paths(max_length=max_complexity, exact_length=exact_complexity)
        t_dur_paths = time.time() - t_start_paths
        logger.info(f"<-- Fetched {len(paths_data) if paths_data else 0} paths in {t_dur_paths:.2f}s ({complexity_mode}).")
        
        if not paths_data:
            logger.warning(f"No paths found ({complexity_mode}).")
            return []

        # --- REMOVED Python filtering --- 
        
        paths_to_process = paths_data
        process_limit_msg = "all available"
        if limit is not None and limit > 0 and limit < len(paths_data):
            logger.info(f"Applying limit: Processing only {limit} of {len(paths_data)} found paths ({complexity_mode}).")
            paths_to_process = paths_data[:limit]
            process_limit_msg = f"{limit}"
        
        process_count = len(paths_to_process)
        if process_count == 0:
            logger.warning(f"No paths to process after limiting ({complexity_mode}).")
            return []
            
        logger.info(f"Processing {process_limit_msg} ({process_count}) paths CONCURRENTLY (max_workers={max_workers}, {complexity_mode}{topic_msg}{reverse_msg}).")

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for path_data in paths_to_process:
                # Pass topic and generate_reverse to single path processor
                future = executor.submit(_process_single_path, path_data, llm_client, topic=topic, generate_reverse=generate_reverse) 
                futures.append(future)

            logger.info(f"Submitted {len(futures)} path processing tasks.")
            processed_count_display = 0 
            log_interval = max(1, len(futures) // 10) 
            for future in as_completed(futures):
                try:
                    result_pair_list = future.result() # This list can have 0, 1, or 2 pairs
                    if result_pair_list:
                        all_generated_pairs.extend(result_pair_list)
                except Exception as e:
                    logger.error(f"A path processing task future failed: {e}", exc_info=False)
                processed_count_display += 1
                if processed_count_display % log_interval == 0 or processed_count_display == len(futures):
                     logger.info(f"Processed {processed_count_display}/{len(futures)} path tasks. Current pairs generated: {len(all_generated_pairs)}")

    except Exception as e:
        logger.error(f"An error occurred during CONCURRENT QA generation from paths: {e}", exc_info=True)

    logger.info(f"Finished generating {len(all_generated_pairs)} QA pairs from paths ({complexity_mode}, Limit Applied: {limit is not None}, Processed Count: {process_count}{topic_msg}{reverse_msg}).")
    return all_generated_pairs 