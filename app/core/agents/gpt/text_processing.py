import re
import logging
import json
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.common.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

_llm_cache = None


def _get_llm():
    """Lazy-initialize LLM so tests can import this module without OPENAI_* env set."""
    global _llm_cache
    if _llm_cache is None:
        if not OPENAI_MODEL:
            raise ValueError("OPENAI_MODEL must be set in environment (e.g. in .env)")
        kwargs = dict(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, max_tokens=2000, temperature=0.1)
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        _llm_cache = ChatOpenAI(**kwargs)
    return _llm_cache

def preprocess_text(text):
    sentences = sent_tokenize(text)
    logger.debug(f"Preprocessed text into {len(sentences)} sentences.")
    return sentences

def extract_triplets_with_gpt(text):
    # Updated System Prompt with refinements
    system_prompt_content = """You are an information-extraction specialist.

YOUR TASK  
Extract only the most significant and meaningful **subject–predicate–object** triplets from any text you receive.  
Return your answer strictly in the JSON schema shown below.

GUIDELINES  
• Focus on important entities: names, places, concepts, achievements.  
• Include defining characteristics and significant relationships.  
• Capture major influences, contributions, and key life events.  
• Skip generic pronouns (he, she, it, they) *after* attempting to resolve them to the specific entity they refer to, if possible.
• **Extract only relationships explicitly stated or directly implied in the text.** Do not infer relationships based on world knowledge not present in the provided text.
• Write relations in clear lowercase_with_underscores.

OUTPUT FORMAT (MANDATORY)  
{
  "triplets": [
    {"head": "specific_entity", "relation": "significant_relation", "tail": "important_concept"},
    {"head": "major_figure", "relation": "notable_achievement", "tail": "specific_contribution"}
  ]
}

EXAMPLES (GOOD)  
✓ {"head": "hafiz", "relation": "wrote", "tail": "persian poetry"}  
✓ {"head": "persian poetry", "relation": "explores_themes_of", "tail": "mysticism"}  
✓ {"head": "hafiz", "relation": "influenced", "tail": "sufi literature"}  

AVOID (BAD)  
✗ {"head": "he", "relation": "was", "tail": "there"}  
✗ {"head": "the poet", "relation": "has", "tail": "words"}

**Ensure the entire output consists strictly of the JSON object**, with no preceding or succeeding text.
"""
    
    # New Human Prompt
    human_prompt_content = f"Text: {text}"

    try:
        # Invoke with System and Human messages
        response = _get_llm().invoke([
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=human_prompt_content)
        ]).content
        
        logger.debug(f"GPT response for triplet extraction: {response}")
        response = response.strip()
        
        # Try to find the outermost JSON object
        json_start = response.find('{')
        json_end = response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response[json_start:json_end+1]
            logger.debug(f"Attempting to parse extracted JSON: {json_str}")
            try:
                data = json.loads(json_str)
                triplets = data.get("triplets", [])
                if isinstance(triplets, list):
                    well_formed, malformed = _partition_well_formed(triplets)
                    if malformed:
                        logger.warning(f"Dropped {len(malformed)} malformed triplet(s) of {len(triplets)}: {malformed}")
                    if well_formed:
                        logger.info(f"Successfully extracted {len(well_formed)} triplets via JSON parsing.")
                        return well_formed
                logger.warning("Parsed JSON structure is invalid. Falling back to regex.")
                # Fallback to regex on the original response might be better than regex on bad JSON
                return extract_triplets_from_text(response)
            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse extracted JSON: {json_err}. Falling back to regex.")
                return extract_triplets_from_text(response)
        else:
             logger.warning("Could not find valid JSON object markers '{' and '}' in LLM response. Falling back to regex.")
             return extract_triplets_from_text(response)
             
    except Exception as e:
        logger.error(f"Error during GPT triplet extraction process: {e}")
        return [] # Return empty list on major failure

_TRIPLET_KEYS = ("head", "relation", "tail")

def _partition_well_formed(triplets):
    """Split parsed triplets into (usable, malformed) — see `_is_well_formed`."""
    well_formed, malformed = [], []
    for triplet in triplets:
        (well_formed if _is_well_formed(triplet) else malformed).append(triplet)
    return well_formed, malformed

def _is_well_formed(triplet):
    """True if head, relation and tail are all present AND non-empty strings.

    Checking that the *keys* exist is not enough: the agent model emits `"tail": null`
    often enough to matter (3 times on a single golden_gate_bridge run), which satisfies
    `'tail' in t` and then raises inside `clean_triplet` on `None.strip()`. That
    exception escapes `extract_clean_special_terms` entirely, so one null discards every
    well-formed triplet pooled from every batch of the same text — and on the seed path
    it aborts `generate_response`, leaving the run with no graph at all. Dropping the one
    bad triplet here keeps its siblings, and keeps which branches survive a property of
    the text rather than of the model's luck in that batch.
    """
    if not isinstance(triplet, dict):
        return False
    return all(
        isinstance(triplet.get(key), str) and triplet[key].strip() for key in _TRIPLET_KEYS
    )

def extract_triplets_from_text(text):
    triplets = []
    pattern = r"[\"']head[\"']\s*:\s*[\"']([^\"']+)[\"']\s*,\s*[\"']relation[\"']\s*:\s*[\"']([^\"']+)[\"']\s*,\s*[\"']tail[\"']\s*:\s*[\"']([^\"']+)[\"']"
    matches = re.finditer(pattern, text)
    for match in matches:
        head, relation, tail = match.groups()
        triplets.append({
            "head": head.strip(),
            "relation": relation.strip(),
            "tail": tail.strip()
        })
    logger.debug(f"Regex extraction found {len(triplets)} triplets")
    return triplets

def remove_redundant_triplets(triplets):
    unique_triplets = []
    seen = set()
    
    def normalize_term(term):
        return re.sub(r'\s+', ' ', term.lower().replace("_", " ")).strip()
    
    for triplet in triplets:
        head = normalize_term(triplet["head"])
        relation = re.sub(r"\W|^(?=\d)", "_", triplet["relation"].strip().lower())
        tail = normalize_term(triplet["tail"])
        term_pair = frozenset([head, tail])
        relation_key = (term_pair, relation)
        if relation_key not in seen:
            seen.add(relation_key)
            triplet["head"] = triplet["head"].strip().lower().replace("_", " ")
            triplet["relation"] = relation
            triplet["tail"] = triplet["tail"].strip().lower().replace("_", " ")
            unique_triplets.append(triplet)
    logger.debug(f"Removed redundant triplets. Unique triplets count: {len(unique_triplets)}")
    return unique_triplets

def clean_triplet(triplet):
    return {
        "head": triplet["head"].strip().lower().replace("_", " "),
        "relation": re.sub(r"\W|^(?=\d)", "_", triplet["relation"].strip().lower()),
        "tail": triplet["tail"].strip().lower().replace("_", " ")
    }

def extract_clean_special_terms(text):
    sentences = preprocess_text(text)
    all_triplets = []
    batch_size = 3
    for i in range(0, len(sentences), batch_size):
        batch_text = " ".join(sentences[i:i + batch_size])
        triplets = extract_triplets_with_gpt(batch_text)
        all_triplets.extend(triplets)
    if not all_triplets:
        all_triplets = extract_triplets_with_gpt(text)
    clean_triplets = [clean_triplet(t) for t in all_triplets]
    unique_triplets = remove_redundant_triplets(clean_triplets)
    logger.info(f"Extracted {len(unique_triplets)} unique triplets")
    return unique_triplets

def extract_triplets_from_response(response):
    return extract_clean_special_terms(response)
