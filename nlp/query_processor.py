"""
Query processing and classification for the Manzil Chatbot.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import json
from openai import OpenAI

from config import settings
from utils.logging_config import get_logger
from utils.error_handlers import ErrorHandler, retry

logger = get_logger("nlp.query_processor")


class QueryProcessor:
    """
    Processes, classifies, and enhances user queries.
    """

    def __init__(self):
        """Initialize the query processor."""
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.COMPLETION_MODEL

        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized query processor with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

        # Load Islamic finance terminology
        self.islamic_terms = {
            "murabaha",
            "ijara",
            "musharaka",
            "sukuk",
            "takaful",
            "riba",
            "gharar",
            "zakat",
            "halal",
            "haram",
            "shariah",
            "sharia",
            "fatwa",
            "wasiyyah",
            "qard",
            "sadaqah",
            "manzil",
            "islamic finance",
            "islamic banking",
            "islamic mortgage",
            "islamic investing",
        }

        # Define Manzil product terms
        self.product_terms = {
            "halal home financing",
            "halal mortgage",
            "murabaha mortgage",
            "halal investing",
            "halal wealth management",
            "islamic will",
            "wasiyyah",
            "halal prepaid card",
            "halal mastercard",
            "manzil communities",
        }

        # Classification patterns
        self.query_patterns = {
            "factual": [
                r"what is",
                r"how does",
                r"when",
                r"where",
                r"who",
                r"which",
                r"explain",
                r"define",
                r"tell me about",
                r"eligibility",
                r"requirements",
                r"qualify",
            ],
            "exploratory": [
                r"how can i",
                r"what options",
                r"help me",
                r"can you help",
                r"i need",
                r"ways to",
                r"i want to",
                r"looking for",
                r"options for",
                r"alternatives",
            ],
            "process": [
                r"how do i",
                r"steps to",
                r"process for",
                r"procedure",
                r"application",
                r"apply for",
                r"get started",
                r"setup",
            ],
            "comparison": [
                r"compare",
                r"versus",
                r"vs",
                r"difference between",
                r"better than",
                r"preferred",
                r"advantage",
                r"disadvantage",
            ],
            "islamic_finance": [
                r"halal",
                r"haram",
                r"islamic",
                r"shariah",
                r"sharia",
                r"riba",
                r"interest",
                r"prohibited",
                r"permitted",
                r"allowed",
            ],
            "out_of_scope": [
                r"12 year",
                r"minor",
                r"child",
                r"illegal",
                r"fraud",
                r"bypass",
                r"hack",
                r"trick",
                r"loophole",
                r"workaround",
            ],
        }

    def process_query(self, query: str) -> Tuple[str, str]:
        """
        Process a user query to determine its type and enhance it.

        Args:
            query (str): The user's question

        Returns:
            tuple: (query_type, enhanced_query)
        """
        with ErrorHandler(error_type="query_processing", reraise=True):
            if not query or query.isspace():
                return "general", ""

            # Clean the query
            cleaned_query = self._clean_query(query)

            # Determine query type
            query_type = self._classify_query(cleaned_query)
            logger.debug(f"Classified query as '{query_type}': {cleaned_query}")

            # Enhance the query
            enhanced_query = self._enhance_query(cleaned_query, query_type)

            return query_type, enhanced_query

    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize a user query.

        Args:
            query (str): The raw query text

        Returns:
            str: Cleaned query
        """
        # Convert to lowercase
        query = query.lower()

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query).strip()

        # Remove special characters but keep apostrophes
        query = re.sub(r"[^\w\s\'.,?!]", "", query)

        return query

    def _classify_query(self, query: str) -> str:
        """
        Classify the query type based on patterns.

        Args:
            query (str): The cleaned query text

        Returns:
            str: Query type
        """
        # First check for Islamic finance terminology
        for term in self.islamic_terms:
            if term in query.lower():
                return "islamic_finance"

        # Check patterns for each query type
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type

        # Default to general if no match
        return "general"

    def _enhance_query(self, query: str, query_type: str) -> str:
        """
        Enhance the query to improve retrieval.

        Args:
            query (str): The cleaned query
            query_type (str): The classified query type

        Returns:
            str: Enhanced query
        """
        # Add domain-specific terms if applicable
        enhanced = query

        # For Islamic finance queries, ensure terminology variations are included
        if query_type == "islamic_finance":
            # Check for terminology variations
            if "sharia" in query and "shariah" not in query:
                enhanced += " shariah"
            elif "shariah" in query and "sharia" not in query:
                enhanced += " sharia"

            # If asking about Islamic finance principles, expand
            if "riba" in query:
                enhanced += " interest prohibition islamic finance"
            elif "halal" in query and "investment" in query:
                enhanced += " shariah compliant investing"

        # For product-specific queries, add synonyms
        for product in self.product_terms:
            if product in query:
                # Add Manzil to product queries if not present
                if "manzil" not in query:
                    enhanced += " manzil"
                break

        # For comparison queries, ensure both options are clear
        if query_type == "comparison" and "vs" in query:
            # Convert "vs" or "versus" to "compared to" for better retrieval
            enhanced = enhanced.replace("vs", "compared to")
            enhanced = enhanced.replace("versus", "compared to")

        logger.debug(f"Enhanced query: {enhanced}")
        return enhanced

    @retry(exceptions=(Exception,), tries=2, delay=1, backoff=2, logger=logger)
    def extract_entities(
        self, query: str, response: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from query and response using OpenAI.

        Args:
            query (str): The user's question
            response (str, optional): The chatbot's response

        Returns:
            list: List of extracted entities with metadata
        """
        with ErrorHandler(
            error_type="entity_extraction", reraise=False, fallback_value=[]
        ):
            # First try rule-based extraction for efficiency
            rule_based_entities = self._rule_based_entity_extraction(query)

            # If we found enough entities, return them
            if len(rule_based_entities) >= 2:
                return rule_based_entities

            # Otherwise, use OpenAI for more sophisticated extraction
            if not response:
                text = query
            else:
                text = f"User: {query}\nAssistant: {response}"

            try:
                system_prompt = """You are an entity extraction assistant for Manzil, a Canadian Islamic finance company. 
Extract entities from the text and classify them into the following types:
- person: Individual people
- organization: Companies, institutions, government bodies
- product: Financial products or services
- topic: Financial concepts or subjects
- location: Geographic locations
- islamic_term: Islamic finance terminology

Return ONLY a JSON array with the following structure for each entity:
[{"name": "entity name", "type": "entity type", "metadata": {"description": "brief description if known"}}]

Only include entities that are clearly mentioned in the text. Focus on Islamic finance terminology when present."""

                messages = [
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": f"Extract entities from: {text}"},
                ]

                response = self.client.responses.create(
                    model=self.model, input=messages
                )

                extracted_text = response.output_text

                # Try to parse as JSON
                try:
                    # Extract JSON array from response if not a clean JSON
                    json_match = re.search(r"\[.*\]", extracted_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                        entities = json.loads(json_text)
                    else:
                        entities = json.loads(extracted_text)

                    logger.debug(f"Extracted {len(entities)} entities using OpenAI")
                    return entities
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse entities JSON: {extracted_text}")
                    return rule_based_entities

            except Exception as e:
                logger.error(f"Error extracting entities: {str(e)}")
                return rule_based_entities

    def _rule_based_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using rule-based approaches.

        Args:
            text (str): The text to extract entities from

        Returns:
            list: List of extracted entities
        """
        entities = []
        text_lower = text.lower()

        # Check for Manzil
        if "manzil" in text_lower:
            entities.append(
                {
                    "name": "Manzil",
                    "type": "organization",
                    "metadata": {"description": "Canadian Islamic finance company"},
                }
            )

        # Check for Islamic finance terms
        for term in self.islamic_terms:
            if term in text_lower:
                # Convert to proper case for display
                proper_term = term.capitalize()
                entity_type = "islamic_term"

                # Add description based on term
                description = ""
                if term == "murabaha":
                    description = "Cost-plus financing structure in Islamic finance"
                elif term == "ijara":
                    description = "Islamic leasing arrangement"
                elif term == "sukuk":
                    description = "Islamic financial certificates (similar to bonds)"
                elif term == "takaful":
                    description = "Islamic insurance concept"
                elif term == "riba":
                    description = "Interest, which is prohibited in Islamic finance"
                elif term == "shariah" or term == "sharia":
                    description = "Islamic law derived from the Quran and Sunnah"
                elif term == "zakat":
                    description = "Islamic form of almsgiving or charity"
                elif term == "wasiyyah":
                    description = "Islamic will"

                entities.append(
                    {
                        "name": proper_term,
                        "type": entity_type,
                        "metadata": {"description": description},
                    }
                )

        # Check for Manzil products
        for product in self.product_terms:
            if product in text_lower:
                # Convert to proper case for display
                proper_product = " ".join(word.capitalize() for word in product.split())

                entities.append(
                    {"name": proper_product, "type": "product", "metadata": {}}
                )

        # Deduplicate entities by name
        unique_entities = []
        seen = set()
        for entity in entities:
            if entity["name"].lower() not in seen:
                seen.add(entity["name"].lower())
                unique_entities.append(entity)

        return unique_entities

    def is_clarification_needed(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if query needs clarification.

        Args:
            query (str): The user's question

        Returns:
            tuple: (needs_clarification, ambiguity_type)
        """
        # Check for very short queries
        if len(query.split()) < 3:
            return True, "general"

        # Check for overly general product inquiries
        if "product" in query.lower() and not any(
            term in query.lower() for term in self.product_terms
        ):
            return True, "product"

        # Check for ambiguous service inquiries
        if "service" in query.lower() and not any(
            term in query.lower() for term in self.product_terms
        ):
            return True, "service"

        # Check for ambiguous Islamic finance terminology
        islamic_term_count = sum(
            1 for term in self.islamic_terms if term in query.lower()
        )
        if islamic_term_count > 2:
            return True, "term"

        return False, None

    def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """
        Generate relevant follow-up questions based on the conversation.

        Args:
            query (str): The user's question
            response (str): The chatbot's response

        Returns:
            list: List of suggested follow-up questions
        """
        with ErrorHandler(
            error_type="follow_up_generation", reraise=False, fallback_value=[]
        ):
            try:
                system_prompt = """You are a follow-up question generator for Manzil, a Canadian Islamic finance company.
Based on the conversation, suggest 2-3 natural follow-up questions that the user might want to ask next.
Focus on Islamic finance topics, product details, application processes, and eligibility requirements.
Return ONLY a JSON array of questions with no additional text."""

                messages = [
                    {"role": "developer", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"User: {query}\nAssistant: {response}",
                    },
                ]

                response = self.client.responses.create(
                    model=self.model, input=messages
                )

                generated_text = response.output_text

                # Try to parse as JSON
                try:
                    # Extract JSON array from response if not a clean JSON
                    json_match = re.search(r"\[.*\]", generated_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                        questions = json.loads(json_text)
                    else:
                        questions = json.loads(generated_text)

                    # Ensure we have a list of strings
                    if isinstance(questions, list) and all(
                        isinstance(q, str) for q in questions
                    ):
                        return questions[:3]  # Limit to top 3
                    else:
                        # Handle case where we got objects instead of strings
                        return [q for q in questions if isinstance(q, str)][:3]

                except json.JSONDecodeError:
                    # If JSON parsing fails, extract questions using regex
                    questions = re.findall(r'"([^"]+\?)"', generated_text)
                    if not questions:
                        questions = re.findall(
                            r"(?:^|\n)(?:\d+\.\s*)?(.+\?)", generated_text
                        )
                    return questions[:3]  # Limit to top 3

            except Exception as e:
                logger.error(f"Error generating follow-up questions: {str(e)}")
                return []
