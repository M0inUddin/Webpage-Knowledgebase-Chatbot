"""
Configuration settings for the Manzil Chatbot application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-4.1")

# Website crawling settings
BASE_URL = os.getenv("BASE_URL", "https://help.manzil.ca/hc/en-us")
MAX_PAGES = int(os.getenv("MAX_PAGES", "100"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))  # seconds
RESPECT_ROBOTS_TXT = os.getenv("RESPECT_ROBOTS_TXT", "True").lower() == "true"
USER_AGENT = os.getenv("USER_AGENT", "ManzilHelpChatbot/1.0")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds

# Content processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

# Database settings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "manzil_help_content")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
ENABLE_KNOWLEDGE_GRAPH = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "False").lower() == "true"

# RAG settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100000"))

# Performance settings
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))  # requests per minute
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# Application settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")

# Domain-specific settings for Manzil
COMPANY_NAME = "Manzil"
COMPANY_DESCRIPTION = "A Canadian financial services company dedicated to providing Shariah-compliant financial solutions."
SERVICES = [
    "Halal Home Financing (Murabaha mortgage)",
    "Halal Investing & Wealth Management",
    "Islamic Wills (Wasiyyah)",
    "Halal Prepaid MasterCard",
    "Manzil Communities (residential projects)",
    "Zakat Calculation",
    "Realty Services",
    "Financial Education",
]


# System checks
def validate_settings():
    """Validate critical settings and return a list of any issues."""
    issues = []

    if not OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY is not set. Please provide an API key.")

    if MAX_PAGES <= 0:
        issues.append("MAX_PAGES must be greater than 0.")

    if CHUNK_SIZE <= CHUNK_OVERLAP:
        issues.append("CHUNK_SIZE must be greater than CHUNK_OVERLAP.")

    return issues


def get_config_summary():
    """Return a summary of the current configuration."""
    return {
        "base_url": BASE_URL,
        "max_pages": MAX_PAGES,
        "embedding_model": EMBEDDING_MODEL,
        "completion_model": COMPLETION_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection_name": COLLECTION_NAME,
        "debug_mode": DEBUG_MODE,
        "enable_knowledge_graph": ENABLE_KNOWLEDGE_GRAPH,
        "company_name": COMPANY_NAME,
    }
