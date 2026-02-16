from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# For GPT agent, fallback to generic values if GPT-specific ones are not provided
GPT_NEO4J_URI = os.getenv("GPT_NEO4J_URI")
GPT_NEO4J_USER = os.getenv("GPT_NEO4J_USER")
GPT_NEO4J_PASSWORD = os.getenv("GPT_NEO4J_PASSWORD")

# For REBEL agent
REBEL_NEO4J_URI = os.getenv("REBEL_NEO4J_URI")
REBEL_NEO4J_USER = os.getenv("REBEL_NEO4J_USER")
REBEL_NEO4J_PASSWORD = os.getenv("REBEL_NEO4J_PASSWORD")

# LLM: any OpenAI-compatible API (set OPENAI_API_BASE if using a different endpoint)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or None  # None = use OpenAI default

# Default maximum recursion depth for triplet extraction
MAX_DEPTH = 2

# Default Descriptions
DEFAULT_NO_DESCRIPTION = "Description not available."
DEFAULT_ERROR_DESCRIPTION = "I'm sorry, I encountered an error while processing your request."
