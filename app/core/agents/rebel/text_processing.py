import re
import logging
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

try:
    rebel_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    rebel_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    rebel_model.to("cpu")
    logger.info("REBEL model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading REBEL model: {e}")
    raise RuntimeError(f"Error loading REBEL model: {e}")

gen_kwargs = {
    "max_length": 128,
    "num_beams": 4,
    "num_return_sequences": 1,
    "length_penalty": 1.0,
    "top_p": 1.0,
    "temperature": 1.0,
    "do_sample": False,
}

def preprocess_text(text):
    sentences = sent_tokenize(text)
    logger.debug(f"Preprocessed text into {len(sentences)} sentences.")
    return sentences

def extract_triplets_from_text(text):
    triplets = []
    text = text.strip()
    logger.debug(f"Raw REBEL output: {text}")
    try:
        data = json.loads(text)
        for item in data.get("triplets", []):
            head = item.get("head", "").strip()
            relation = item.get("relation", "").strip()
            tail = item.get("tail", "").strip()
            if head and relation and tail:
                triplets.append({"head": head, "relation": relation, "tail": tail})
        logger.debug(f"Extracted {len(triplets)} triplets from JSON.")
        return triplets
    except json.JSONDecodeError:
        logger.debug("JSON parsing failed. Using marker-based extraction.")
    relation, subject, object_ = "", "", ""
    current = None
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            if subject and relation and object_:
                triplets.append({"head": subject.strip(), "relation": relation.strip(), "tail": object_.strip()})
            subject, relation, object_ = "", "", ""
            current = "subject"
        elif token == "<subj>":
            current = "object"
            if subject and relation and object_:
                triplets.append({"head": subject.strip(), "relation": relation.strip(), "tail": object_.strip()})
            object_ = ""
        elif token == "<obj>":
            current = "relation"
            relation = ""
        else:
            if current == "subject":
                subject += " " + token
            elif current == "object":
                object_ += " " + token
            elif current == "relation":
                relation += " " + token
    if subject and relation and object_:
        triplets.append({"head": subject.strip(), "relation": relation.strip(), "tail": object_.strip()})
    logger.debug(f"Extracted {len(triplets)} triplets using marker-based parsing.")
    return triplets

def extract_triplets(text):
    """
    Extract triplets from the input text using the REBEL model.
    Includes a post-validation step to ensure entities exist in the source sentence.
    """
    validated_triplets = [] # Store triplets that pass validation
    sentences = preprocess_text(text)

    for sentence in sentences:
        sentence_lower = sentence.lower() # Lowercase sentence for validation check
        if not sentence.strip():
            continue

        model_inputs = rebel_tokenizer(
            sentence,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(rebel_model.device) # Ensure inputs are on the same device as the model

        try:
            generated_tokens = rebel_model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                **gen_kwargs,
            )
            decoded_preds = rebel_tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=False
            )

            sentence_triplets_raw = []
            for pred in decoded_preds:
                logger.debug(f"Raw REBEL prediction for '{sentence[:50]}...': {pred}")
                extracted = extract_triplets_from_text(pred) # Parse the <triplet> format
                sentence_triplets_raw.extend(extracted)
            
            # Validate extracted triplets against the source sentence
            for triplet in sentence_triplets_raw:
                head_lower = triplet["head"].lower().strip()
                tail_lower = triplet["tail"].lower().strip()
                
                # Check if both head and tail appear in the original sentence (case-insensitive)
                if head_lower in sentence_lower and tail_lower in sentence_lower:
                    validated_triplets.append(triplet) # Keep the triplet
                    logger.debug(f"Validated triplet: {triplet} (Source: '{sentence[:50]}...')")
                else:
                    logger.debug(f"Discarded triplet (entities not in source): {triplet} (Source: '{sentence[:50]}...')")

        except Exception as e:
            logger.error(f"Error during triplet generation/validation for sentence '{sentence[:50]}...': {e}", exc_info=True)
            continue

    logger.debug(f"Total validated triplets before cleaning: {len(validated_triplets)}")
    return validated_triplets # Return only validated triplets

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
    triplets = extract_triplets(text)
    clean_triplets = [clean_triplet(t) for t in triplets]
    unique_triplets = remove_redundant_triplets(clean_triplets)
    logger.info(f"Extracted {len(unique_triplets)} unique triplets")
    return unique_triplets
