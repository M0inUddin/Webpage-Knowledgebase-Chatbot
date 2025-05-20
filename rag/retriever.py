"""
Context retrieval for the Manzil Chatbot RAG pipeline.
"""

from typing import List, Dict, Any, Optional
import re

from config import settings
from utils.logging_config import get_logger
from utils.error_handlers import ErrorHandler
from knowledge_base.vector_store import VectorStore

logger = get_logger("rag.retriever")


class Retriever:
    """
    Retrieves relevant context for user queries from the knowledge base.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the retriever.

        Args:
            vector_store (VectorStore): The vector store for retrieving documents
        """
        self.vector_store = vector_store
        self.top_k = settings.TOP_K_RESULTS

        # Initialize with Islamic finance keywords for domain-specific retrieval
        self.islamic_keywords = {
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
        }

        # Manzil product keywords
        self.product_keywords = {
            "mortgage",
            "financing",
            "invest",
            "investment",
            "will",
            "wasiyyah",
            "prepaid",
            "card",
            "mastercard",
            "communities",
        }

    def retrieve(
        self, query: str, query_type: str = "general", top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query (str): The user's question or enhanced query
            query_type (str): The type of query
            top_k (int, optional): Number of results to retrieve

        Returns:
            list: List of retrieved documents with metadata
        """
        with ErrorHandler(error_type="retrieval", reraise=True):
            # Use provided top_k or default
            k = top_k if top_k is not None else self.top_k

            # Adjust k based on query type
            if query_type == "exploratory":
                # More documents for exploratory queries
                k = max(k, 4)
            elif query_type == "comparison":
                # More documents for comparison queries
                k = max(k, 4)

            # Check if query contains Islamic finance terms
            has_islamic_terms = any(
                term in query.lower() for term in self.islamic_keywords
            )

            # Check if query contains product terms
            has_product_terms = any(
                term in query.lower() for term in self.product_keywords
            )

            # Retrieve documents using vector similarity
            retrieved_docs = self.vector_store.retrieve_similar(query, k)

            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: {query}")
                return []

            # Post-process and rank retrieved documents
            processed_docs = self._post_process_results(
                retrieved_docs, query, has_islamic_terms, has_product_terms
            )

            # Log retrieval results
            logger.info(
                f"Retrieved {len(processed_docs)} documents for query: {query[:50]}..."
            )
            for i, doc in enumerate(processed_docs):
                title = doc.get("metadata", {}).get(
                    "document_title", doc.get("metadata", {}).get("title", "Untitled")
                )
                logger.debug(f"  Doc {i+1}: {title} (score: {doc.get('score', 0):.4f})")

            return processed_docs

    def _post_process_results(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        has_islamic_terms: bool,
        has_product_terms: bool,
    ) -> List[Dict[str, Any]]:
        """
        Post-process and re-rank retrieved documents.

        Args:
            docs (list): Retrieved documents
            query (str): The original query
            has_islamic_terms (bool): Whether the query contains Islamic finance terms
            has_product_terms (bool): Whether the query contains product terms

        Returns:
            list: Processed and re-ranked documents
        """
        if not docs:
            return []

        # Convert query to lowercase for matching
        query_lower = query.lower()

        # Extract query keywords for exact matching
        keywords = self._extract_keywords(query)

        # Score and re-rank documents
        for doc in docs:
            content = doc["content"].lower()
            metadata = doc.get("metadata", {})

            # Start with base score (inverse of distance from vector search)
            base_score = 1.0 - (doc.get("distance", 0) * 0.5)

            # Adjust score based on content relevance
            relevance_score = 0.0

            # Check for exact phrase matches (highest value)
            if query_lower in content:
                relevance_score += 0.3

            # Check for keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in content)
            relevance_score += 0.05 * keyword_matches

            # Boost for domain relevance
            if has_islamic_terms and any(
                term in content for term in self.islamic_keywords
            ):
                relevance_score += 0.2

            if has_product_terms and any(
                term in content for term in self.product_keywords
            ):
                relevance_score += 0.2

            # Boost for metadata relevance (section titles, document titles)
            if (
                "document_title" in metadata
                and query_lower in metadata["document_title"].lower()
            ):
                relevance_score += 0.15

            if (
                "section_title" in metadata
                and query_lower in metadata["section_title"].lower()
            ):
                relevance_score += 0.2

            # Calculate final score
            final_score = base_score + relevance_score

            # Add score to document
            doc["score"] = min(1.0, final_score)  # Cap at 1.0

        # Sort by score in descending order
        docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        return docs

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text (str): The text to extract keywords from

        Returns:
            list: List of keywords
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)

        # Split into words
        words = text.split()

        # Remove common stop words
        stop_words = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
        }

        # Filter out stop words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Add multi-word phrases
        phrases = []
        for i in range(len(words) - 1):
            phrase = words[i] + " " + words[i + 1]
            phrases.append(phrase)

        # Combine and deduplicate
        all_keywords = list(set(keywords + phrases))

        return all_keywords

    def retrieve_by_topic(self, topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents related to a specific topic.

        Args:
            topic (str): The topic to retrieve documents for
            top_k (int): Number of results to retrieve

        Returns:
            list: List of retrieved documents with metadata
        """
        with ErrorHandler(
            error_type="topic_retrieval", reraise=False, fallback_value=[]
        ):
            # Construct topic-specific query
            topic_query = f"information about {topic} in Islamic finance"

            # Retrieve documents
            docs = self.vector_store.retrieve_similar(topic_query, top_k)

            # Post-process for topic relevance
            for doc in docs:
                # Check for topic presence in content
                content = doc["content"].lower()
                if topic.lower() in content:
                    doc["score"] = doc.get("score", 0.5) + 0.2

            # Sort by score
            docs.sort(key=lambda x: x.get("score", 0), reverse=True)

            return docs

    def retrieve_by_metadata(
        self, metadata_filter: Dict[str, Any], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents by metadata filtering.

        Args:
            metadata_filter (dict): Filter criteria for metadata
            limit (int): Maximum number of results to return

        Returns:
            list: List of matching documents with metadata
        """
        with ErrorHandler(
            error_type="metadata_retrieval", reraise=False, fallback_value=[]
        ):
            return self.vector_store.search_by_metadata(metadata_filter, limit)
