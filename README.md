# KNIGHT: Knowledge Graph-Driven Multiple-Choice Question Generation with Adaptive Hardness Calibration

[![PyPI version](https://img.shields.io/pypi/v/knight-mcq.svg)](https://pypi.org/project/knight-mcq/) [![Python 3.11+](https://img.shields.io/pypi/pyversions/knight-mcq.svg)](https://pypi.org/project/knight-mcq/)

**Install:** `pip install knight-mcq` · **Package:** [knight-mcq on PyPI](https://pypi.org/project/knight-mcq/)

This repository is the **reference implementation** for the paper **KNIGHT** (CPAL 2026). It builds a topic-specific, reusable knowledge graph from external sources and generates difficulty-controlled multiple-choice question (MCQ) datasets from graph paths, with optional LLM validation.

**Paper:** [KNIGHT on OpenReview (CPAL 2026)](https://openreview.net/forum?id=8kA9oO5gEc)

**Citation:** If you use this code or the paper, please cite using the BibTeX in [CITATION.bib](CITATION.bib). Example:

```bibtex
@inproceedings{knight2026cpal,
  title = {{KNIGHT}: Knowledge Graph-Driven Multiple-Choice Question Generation with Adaptive Hardness Calibration},
  author = {Amanlou, Mohammad and {Shafiee Moghaddam}, Erfan and Nouri, Mahdi and {Amou Jafary}, Yasaman and Farsi, Farhan and Bahrak, Behnam},
  booktitle = {Proceedings of the Conference on Parsing and Linguistic Theories (CPAL)},
  year = {2026},
  url = {https://openreview.net/forum?id=8kA9oO5gEc},
}
```

The system constructs a dynamic KG from conversational interactions and LLM outputs, then synthesizes QA pairs of varying complexity (including multi-hop) from the graph. The default instantiation uses Wikipedia/Wikidata for term descriptions. KNIGHT is **model-agnostic** (default setup uses OpenAI) and **external knowledge is replaceable** (default source is Wikipedia; **PDF files and URLs are also supported**; custom sources can be plugged in via the `ExternalKnowledgeLookup` interface).

---

## Overview

KNIGHT constructs and maintains a topic-specific knowledge graph by processing natural language queries. It uses an LLM (configurable; any LangChain-compatible chat model can be used) to generate responses and extract structured knowledge (triplets) into a Neo4j graph. The KG is then reused to synthesize multiple-choice question/answer pairs of varying complexity (including multi-hop) from graph paths, using node descriptions augmented by external knowledge (default: Wikipedia). To use a custom external source, implement the `ExternalKnowledgeLookup` interface and pass it into term description generation.

**External Knowledge Sources:** KNIGHT supports multiple external knowledge sources:
- **Wikipedia** (default): Automatic lookup via the `WikipediaLookup` class.
- **PDF files**: Use `PDFLookup(pdf_path_or_url)` to load a PDF (local file path or URL). The PDF text is extracted, chunked, and relevant passages are found using LLM-based relevance checks.
- **Custom sources**: Implement the `ExternalKnowledgeLookup` protocol for other sources (e.g. databases, APIs, text files).

Example: Using a PDF instead of Wikipedia:
```python
from app.core.utils.external_knowledge import PDFLookup
from app.core.agents.gpt.term_description import generate_term_description

# Create PDF lookup (supports local path or URL)
pdf_lookup = PDFLookup("path/to/document.pdf")  # or "https://example.com/doc.pdf"

# Use it when generating term descriptions
description, used_external = generate_term_description(
    llm=your_llm,
    term="some term",
    external_lookup=pdf_lookup
)
```

Key features include:

- **Dual Agent Implementations:**
  - **GPT Agent:** Primarily uses carefully crafted prompts to instruct the LLM to directly extract knowledge triplets (`subject-predicate-object`) from text.
  - **REBEL Agent:** Uses a dedicated transformer model (originally designed for relation extraction, adapted here via `text_processing.py`) to identify triplets, followed by an **LLM-based validation step** to verify triplet accuracy against the source text.
- **External Knowledge Integration & Node Descriptions:** Supports Wikipedia (default), PDF files (local or URL), and custom sources via `ExternalKnowledgeLookup`. The LLM generates node descriptions using external knowledge context when available (`wiki_fact_checked='Yes'`), or falls back to structured prompts with source/relationship context (`wiki_fact_checked='No'`).
- **Ambiguity Resolution:** If Wikipedia lookup yields ambiguous results, the LLM uses the original conversation context (when generating the description without Wikipedia) to improve disambiguation.
- **Fact-Checking Status:** Tracks whether the LLM description generation was primarily informed by Wikipedia (`'Yes'`) or by its internal knowledge guided by the structured prompt and source context (`'No'`).
- **Robust Neo4j Storage:** Optimized for storing and managing nodes (`Term` label) and relationships, handling normalization and preventing duplicate relationship creation.
- **Configurable Recursive Exploration:** Allows graph traversal based on extracted triplets, with controls for maximum depth and branching factor to manage exploration scope.
- **Detailed Logging:** Provides separate logs for each agent (`gpt_agent`, `rebel_agent`) within the `logs/chatbot.log` file (with rotation) for easier debugging and tracing.
- **Error Handling & Retries:** Uses `tenacity` for robust handling of transient errors during API calls (LLM, Wikipedia) and database operations.
- **Knowledge Graph QA Generation:** Automatically creates Question/Answer pairs directly from the relationships stored in the knowledge graph. This allows testing understanding and generating training data based on the verified connections within the graph.

A knowledge graph is a structured representation where entities (terms, concepts) become nodes and their relationships become edges. This chatbot dynamically builds and expands this graph based on user interactions and LLM-generated insights.

---

## File structure

```
.
├── .env.example                # Template for environment variables (copy to .env)
├── .gitignore
├── CITATION.bib                # BibTeX for citing the paper
├── LICENSE                     # MIT License
├── pyproject.toml              # Project configuration and dependencies (uv)
├── README.md                   # This file
├── uv.lock                     # Lock file for reproducible installs
├── app/
│   ├── core/
│   │   ├── agents/
│   │   │   ├── gpt/            # GPT Agent
│   │   │   │   ├── chatbot.py          # LLM interaction, triplet processing, Neo4j ops
│   │   │   │   ├── term_description.py # Term description generation (external-knowledge lookup, default Wikipedia)
│   │   │   │   └── text_processing.py  # Triplet extraction via LLM prompting
│   │   │   └── rebel/          # REBEL Agent
│   │   │       ├── chatbot.py          # LLM interaction, triplet processing, LLM validation, Neo4j ops
│   │   │       ├── term_description.py # Term description generation (external-knowledge lookup, default Wikipedia)
│   │   │       └── text_processing.py  # Triplet extraction (e.g. REBEL model logic)
│   │   ├── common/
│   │   │   ├── check_connection.py   # Connection checks
│   │   │   ├── config.py             # Environment variables and constants
│   │   │   └── neo4j_connection.py  # Neo4j connection handler
│   │   └── utils/
│   │       ├── external_knowledge.py # External-knowledge lookup interface (Wikipedia, PDF, custom)
│   │       ├── graph_utils.py        # KG utilities (e.g. prune descriptions)
│   │       └── wikipedia_lookup.py   # Wikipedia search, LLM relevance check, content fetching
│   └── generation/
│       ├── qa_generation.py    # QA pair generation from graph paths
│       └── __init__.py
├── logs/                       # Log files (e.g. chatbot.log)
└── app/tests/                  # Test suite (sanity + integration)
```

---

## 🚀 Features

- **Dual Agent Approaches:**
  - _GPT Agent:_ LLM-prompt based triplet extraction, primarily targeting structured JSON output with a regex fallback mechanism.
  - _REBEL Agent:_ Model-based extraction + LLM-based validation.
- **Knowledge Enrichment & Description Synthesis:**
  - Looks up terms on Wikipedia and uses LLM to check relevance.
  - The LLM synthesizes the final node description, prioritizing relevant Wikipedia summary context (`wiki_fact_checked='Yes'`).
  - If Wikipedia context is unavailable/ambiguous, **both agents** use a detailed, 8-point scientific prompt structure (Definition/Scope, Domains, Subfields, Concepts, Applications, Examples, Related Terms, Research Trends) to generate the description. This prompt incorporates original source text or parent term context to guide the generation. The node gets `wiki_fact_checked='No'`.
- **Robust Ambiguity Handling:**
  - Detects ambiguous Wikipedia results.
  - Provides source text/relationship context to the LLM during description synthesis (especially when Wikipedia context isn't used) to aid disambiguation.
  - Tracks `wiki_fact_checked` status ('Yes'/'No') based on the primary context used by the LLM for generation.
- **Neo4j Knowledge Graph:**
  - Stores terms as nodes with descriptions.
  - Creates typed relationships based on extracted/validated triplets.
  - Dynamic querying and depth-controlled exploration.
- **Performance & Reliability:**
  - Global tracking of processed descriptions per session to reduce redundancy.
  * Per-query tracking of processed terms to avoid duplicate node saving/logging within concurrent operations.
  - Parallel processing (`ThreadPoolExecutor`) for triplet processing, validation, and sub-triplet exploration.
  - Automatic retries for API calls and DB operations.
- **Enhanced Logging:**
  - Named loggers (`gpt_agent`, `rebel_agent`) distinguish agent output.
  - Logs saved to `logs/chatbot.log` with file rotation.
- **Automated QA Generation from Knowledge Graph:**
  - Generates relevant Question/Answer pairs by analyzing multi-step paths within the Neo4j graph (e.g., (Term A)-[:REL_1]->(Term B)-[:REL_2]->(Term C)).
  - Allows configuring the _complexity_ (number of steps/relationships) for generated questions.
  - Includes an optional _LLM validation_ step to check generated Q&A for clarity, correctness based on the path, and relevance to a specific topic.
  - Can generate _reverse questions_ where the answer is the starting point of the path, providing different perspectives.
  - Outputs validated Q&A pairs to a file (e.g., `generated_qa_pairs.json`) for review or use.
- **Knowledge Graph Curation:**
  - Provides a utility to prune descriptions from nodes that were not fact-checked against Wikipedia (i.e., `wiki_fact_checked='No'`), allowing for manual quality control of the graph's descriptive content.

---

### Knowledge Graph QA Generation Workflow

This feature allows the chatbot to automatically generate Question/Answer pairs directly from the structure of the knowledge graph it has built. This is useful for creating evaluation datasets, flashcards, or simply exploring the graph's content in a new way. The process, primarily handled by `app/generation/qa_generation.py`, follows these steps:

1.  **Initiation:** The user triggers the process via the `/generate_qa` command in the chat interface.
2.  **Configuration:** The user interactively provides settings:
    - **Complexity:** Specifies whether to use paths of an _exact_ length (number of relationships) or paths _up to_ a maximum length.
    - **Limit (Optional):** Sets a maximum number of paths to fetch from the graph, useful for managing processing time and cost on large graphs.
    - **Validation:** Determines if the generated Q&A pairs should be checked for quality. If enabled, the user also sets a _sample rate_ (0.0 to 1.0) to validate only a portion or all generated pairs.
    - **Topic Focus (Optional):** If a session topic was set, it's used during generation and validation to keep Q&A relevant.
    - **Reverse QA:** Option to generate additional questions where the _start_ node of the path is the answer, alongside the standard questions where the _end_ node is often the answer.
3.  **Path Finding:** The system queries the Neo4j database (`neo4j_connection.find_paths`) to find paths matching the specified complexity criteria (e.g., `MATCH p=(:Term)-[*2]->(:Term)` for exact complexity 2). It retrieves the nodes (including names and descriptions) and relationship types for each path.
4.  **Concurrent Path Processing:** To speed up generation, the system processes the fetched paths in parallel using multiple threads (`ThreadPoolExecutor`). Each path is handled independently.
5.  **QA Pair Generation per Path (`_process_single_path`):**
    - For each path, specialized prompts (`_format_multihop_qa_prompt`) are constructed. These prompts provide the LLM with the path structure (e.g., `(Start)-[:REL]->(Middle)-[:REL]->(End)`), the descriptions of the start and end nodes, and instructions to formulate a question based on the multi-step relationship, aiming for the end node as the answer.
    - If _Reverse QA_ is enabled, a separate prompt (`_format_multihop_qa_prompt_reverse`) is used to generate a question where the _start_ node is the intended answer.
    - The LLM is called using a safe wrapper (`safe_generate`) that includes timeouts and retries to handle potential API issues.
    - The LLM's response (expected in "Question: ... Answer: ..." format) is parsed.
    - Each successfully generated Q&A pair is assigned a unique ID (`uuid`) and stored with metadata about its source path, complexity, etc.
6.  **Validation (`validate_qa_pairs`):**
    - If validation was enabled, the generated pairs (or a sample based on the rate) are evaluated.
    - **Basic Checks:** Question/Answer length and structure are verified.
    - **LLM Validation:** A separate LLM call uses another specialized prompt (`_format_combined_validation_prompt`) to assess:
      - _Grammar/Fluency:_ Is the question well-formed?
      - _Answerability:_ Can the answer be reasonably inferred _only_ from the provided path structure/details?
      - _Topic Relevance:_ (If a topic was provided) Is the Q&A relevant to the topic?
    - Pairs failing validation are logged and discarded.
7.  **Saving Results:** The final set of validated Q&A pairs is saved to a JSON file (`generated_qa_pairs.json` by default) in the project's root directory.

---

## Requirements and Configuration

- **Software:** Python 3.11+, Neo4j (4.x or 5.x), access to an OpenAI-compatible API (or another LLM provider). KNIGHT is model-agnostic; the default setup uses OpenAI. To use another LLM, instantiate your LangChain chat model and pass it where the code expects an LLM.
- **Dependencies:** Managed by `pyproject.toml` / `uv.lock` (e.g. `langchain-openai`, `neo4j`, `tenacity`, `wikipedia`, `wikipedia-api`; see `pyproject.toml` for the full list).
- **Environment:** Copy `.env.example` to `.env` in the project root and fill in your values. Required/optional variables:

  | Variable | Description |
  |----------|-------------|
  | `OPENAI_API_KEY` | Required for default LLM. |
  | `OPENAI_MODEL` | Model name (e.g. `gpt-4o`, `gpt-4`). |
  | `OPENAI_API_BASE` | Optional; base URL for OpenAI-compatible API (e.g. custom endpoint). |
  | `GPT_NEO4J_URI`, `GPT_NEO4J_USER`, `GPT_NEO4J_PASSWORD` | Neo4j for GPT agent. |
  | `REBEL_NEO4J_URI`, `REBEL_NEO4J_USER`, `REBEL_NEO4J_PASSWORD` | Neo4j for REBEL agent (can match GPT). |
  | `MAX_DEPTH` | Optional; default 2. |

  Example `.env`:

  ```env
  GPT_NEO4J_URI=bolt://localhost:7687
  GPT_NEO4J_USER=neo4j
  GPT_NEO4J_PASSWORD=your-password
  REBEL_NEO4J_URI=bolt://localhost:7687
  REBEL_NEO4J_USER=neo4j
  REBEL_NEO4J_PASSWORD=your-password
  OPENAI_API_KEY=your-openai-api-key
  OPENAI_MODEL=gpt-4o
  ```

  Ensure your Neo4j instance is running.

---

## Installation

**Recommended (PyPI):**

```bash
pip install knight-mcq
```

**For REBEL agent** (requires ML dependencies):
```bash
pip install knight-mcq[ml]
```

Then copy the env template and configure (see [Requirements and Configuration](#requirements-and-configuration)):

```bash
# From the folder where you run the app, or clone the repo just to get .env.example
curl -O https://raw.githubusercontent.com/ErfanShm/knight-mcq/main/.env.example
# Rename to .env and fill in your values
```

**Development / from source:** Clone [the repo](https://github.com/ErfanShm/knight-mcq), then `uv sync` (or `pip install -e .`). Use this if you need to modify the code.

---

## Launch

After installing and configuring `.env`:

- **GPT Agent:** `python -m app.core.agents.gpt.chatbot` (works with base installation)
- **REBEL Agent:** `python -m app.core.agents.rebel.chatbot` (requires `pip install knight-mcq[ml]`)

---

## Reproducing results / Quick start

The paper uses Wikipedia/Wikidata and the pipeline: build a topic-specific KG from queries, then generate difficulty-controlled MCQs from graph paths.

1. **Install:** `pip install knight-mcq`
2. **Configure:** Copy [.env.example](https://github.com/ErfanShm/knight-mcq/blob/main/.env.example) to `.env`; set Neo4j credentials and `OPENAI_API_KEY` (and `OPENAI_API_BASE` if using a custom endpoint).
3. **Start Neo4j**, then run: `python -m app.core.agents.gpt.chatbot`
4. Set an optional session topic (e.g. History, Biology, or Mathematics as in the paper).
5. Ask a question to grow the KG; when the graph has enough structure, run `/generate_qa` and choose complexity and options to produce MCQs.

Full reproduction uses the same Neo4j/API setup and topic flow as in the paper.

---

## 📝 Usage Guide

**Setting the Session Topic:** When you first launch the chatbot (e.g., `python -m app.core.agents.gpt.chatbot`), it will prompt you to enter an optional main topic for the session. This topic is primarily used by the `/generate_qa` feature to ensure the automatically created Question/Answer pairs stay relevant to your area of interest. If you don't provide one, QA generation will not be filtered by topic.

Interact with the running chatbot via the command line:

1.  **Set Exploration Depth:** Before each question or command, the chatbot will ask you to set the _maximum exploration depth_ (default is 1, see `.env` section). This controls how many relationship steps (e.g., `TermA -> TermB -> TermC` is depth 2) the chatbot will explore when processing the information _within an LLM's response_ to extract sub-topics (triplets) and build the knowledge graph. A higher depth explores more connections but takes longer and uses more resources. Press Enter to use the default/previous depth or enter a number (e.g., `2`).
2.  **Interact:** After setting the depth, enter your command or question:
    - **Ask a question:** Simply type your query (e.g., `What is Persian literature?`). The chatbot uses the depth set in step 1 for building the graph from the answer.
    - **Generate Q&A from Graph:** Type `/generate_qa`. This starts an interactive process where you'll be prompted to configure settings (like path complexity, limits, validation) to automatically create Question/Answer pairs based on the existing graph structure.
    - **Prune Descriptions:** Type `/prune_descriptions`. This command will ask for confirmation and then set the `description` property to null for all nodes where `wiki_fact_checked` is `'No'`. Use this carefully for cleanup.
    - **Show related terms:** `show related to [term]` (e.g., `show related to hafiz`).
    - **Help:** `help`
    - **Exit:** `bye` or `exit`

---

## ⚙️ Technical Implementation Details

### Knowledge Extraction & Validation

- **GPT Agent:** Relies on a detailed system prompt (`app/core/agents/gpt/text_processing.py`) that instructs the LLM to act as an "information-extraction specialist" and return `subject-predicate-object` triplets in a specific JSON schema. The agent attempts to parse this JSON directly. If parsing fails (e.g., due to minor LLM deviations from the schema), a regex-based fallback mechanism is employed to extract triplets from the raw text response. Quality depends heavily on the LLM's ability to follow instructions for the JSON format.
- **REBEL Agent:** Uses a model-based approach (`text_processing.py`) for initial triplet extraction. Crucially, it then employs an LLM validation step (`validate_triplet_with_llm` in `chatbot.py`) where a separate LLM call verifies if each extracted triplet is directly and accurately stated in the source text. This adds an accuracy layer at the cost of performance.

### Term Description Generation

- Both agents use `term_description.py` which orchestrates the process.
- `wikipedia_lookup.py` searches Wikipedia and uses an LLM relevance check.
- The core function (`generate_term_description`) determines the context for the final LLM call:
  - If a relevant, unambiguous Wikipedia page is found, its summary is the primary context. The node gets `wiki_fact_checked='Yes'`.
  - If Wikipedia lookup fails or is ambiguous, **both agents** use a detailed, 8-point scientific prompt structure (Definition/Scope, Domains, Subfields, Concepts, Applications, Examples, Related Terms, Research Trends) to generate the description. This prompt incorporates original source text or parent term context to guide the generation. The node gets `wiki_fact_checked='No'`.
- The LLM call synthesizes the actual description text based on the provided prompt and context.

### Knowledge Graph Updates

- Nodes (`Term` label) are created/merged using `MERGE` in Neo4j (`save_term_as_node`).
- Descriptions and `wiki_fact_checked` status are added/updated using `SET`.
- Relationships (`type` derived from triplet relation) are created using `MERGE` between existing nodes (`create_relationship`).
- Concurrency control (`current_query_processed_terms` set) prevents duplicate node saving logs during parallel triplet processing.

### Logging

- Uses Python's standard `logging` module.
- Handlers are configured in each agent's `chatbot.py`.
- Named loggers (`gpt_agent`, `rebel_agent`) differentiate output.
- Logs are directed to both the console (INFO level) and a rotating file (`logs/chatbot.log`, DEBUG level).

### Knowledge Graph Curation

- The system includes a utility (`app/core/utils/graph_utils.py`) to prune descriptions from nodes.
- Specifically, the `prune_non_wiki_descriptions` function can be invoked (e.g., via the `/prune_descriptions` command in the GPT agent) to set the `description` property to null for all `Term` nodes where the `wiki_fact_checked` property is 'No'.
- This allows users to selectively remove descriptions that were generated by the LLM without the direct backing of a verified Wikipedia summary, offering a way to manage the overall factuality or source-preference of the descriptions within the KG.

---

## LLM Prompt Usage

The Large Language Model (LLM) is utilized in several distinct ways throughout the application:

1.  **Relation Extraction:**

    - Analyzes user input or text to extract knowledge graph triples (Subject, Predicate, Object), forming the basis of the graph construction. (Primarily in `app/core/agents/gpt/chatbot.py`)

2.  **Wikipedia Relevance Check:**

    - Determines if a candidate Wikipedia page title is semantically relevant for defining a specific term, aiding the node description process. (Implemented in `app/core/utils/wikipedia_lookup.py`)

3.  **Node Description Generation:**

    - The LLM generates the definitive description for every node added to the graph.
    - **Input Context:** The prompt provided to the LLM varies based on the success of the Wikipedia lookup:
      - If a relevant, unambiguous Wikipedia summary is found (`wiki_fact_checked='Yes'`), the LLM uses a simpler prompt incorporating that summary as primary context.
      - If Wikipedia lookup fails or returns ambiguous results (`wiki_fact_checked='No'`), **both agents** use a detailed, 8-point scientific prompt structure (Definition/Scope, Domains, Subfields, Concepts, Applications, Examples, Related Terms, Research Trends) to generate the description. This prompt can incorporate context (like a parent term or source text) to guide the LLM.
    - (Logic resides in `app/core/agents/gpt/term_description.py` and `app/core/agents/rebel/term_description.py`)

4.  **QA Generation (Multi-hop Forward):**

    - Creates a question and answer pair based on the information implied by a multi-step path retrieved from the knowledge graph. (Uses `_format_multihop_qa_prompt` in `app/generation/qa_generation.py`)

5.  **QA Generation (Multi-hop Reverse):**

    - Creates a question based on a multi-step path where the start node of the path is the predefined answer. (Uses `_format_multihop_qa_prompt_reverse` in `app/generation/qa_generation.py`)

6.  **QA Validation:**
    - Evaluates generated Question-Answer pairs for grammatical correctness, logical consistency with the source data (e.g., graph path), and optional topic relevance. (Uses `_format_combined_validation_prompt` in `app/generation/qa_generation.py`)

---

## Testing

Run tests with:

```bash
uv run pytest app/tests/
```

The sanity tests in `app/tests/test_sanity.py` run without external services (no Neo4j or API keys). Full tests (e.g. `test_chatbot.py`) require a running Neo4j instance and `OPENAI_API_KEY`; they may be skipped or fail if those are not available.

## Current status

- Functional GPT and REBEL agents; Neo4j-backed KG; Wikipedia-based (replaceable) external knowledge; difficulty-controlled MCQ generation with optional validation; configurable depth and logging.

---

## Publishing to PyPI (maintainers)

**Option A – GitHub Action (recommended)**  
1. In [PyPI Account](https://pypi.org/manage/account/token/) create an API token (scope: entire account or just this project).  
2. In your repo: **Settings → Secrets and variables → Actions** → New repository secret: name `PYPI_API_TOKEN`, value = the token.  
3. Bump `version` in `pyproject.toml`, commit and push.  
4. Create a **Release** (e.g. tag `v0.1.0`): **Releases → Create a new release** → choose tag `v0.1.0`, publish. The workflow will build and upload to PyPI.

**Option B – Manual**  
1. `pip install build twine` (or `uv sync --extra dev`).  
2. Bump `version` in `pyproject.toml`.  
3. `python -m build` then `twine upload dist/*` (use your PyPI token when prompted). Test first with `twine upload --repository testpypi dist/*` if you prefer.

---

## 🙌 Acknowledgments

- The creators of the REBEL model and other foundational NLP models.
- The OpenAI team and developers of similar large language models.
- The Neo4j team for their graph database technology.
- The developers of the `wikipedia`, `wikipedia-api`, `langchain`, and `tenacity` libraries.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
