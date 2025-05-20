"""
Embedding generation for the Manzil Chatbot using OpenAI's text embeddings.
"""

import time
from typing import List
from openai import OpenAI

from config import settings
from utils.logging_config import get_logger
from utils.error_handlers import EmbeddingException, retry, ErrorHandler, APIException
from utils.rate_limiters import api_limiter

logger = get_logger("nlp.embeddings")


class EmbeddingGenerator:
    """
    Generates embeddings for text using OpenAI's embedding models.
    """

    def __init__(self):
        """Initialize the embedding generator with API settings."""
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.EMBEDDING_MODEL

        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized embedding generator with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise EmbeddingException(
                f"Failed to initialize embedding generator: {str(e)}"
            )

    @retry(exceptions=(Exception,), tries=3, delay=2, backoff=2, logger=logger)
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a single text using OpenAI API.

        Args:
            text (str): The text to embed

        Returns:
            list: Embedding vector as a list of floats
        """
        with ErrorHandler(error_type="embedding", reraise=True):
            if not text or text.isspace():
                logger.warning("Attempted to embed empty text")
                raise ValueError("Cannot embed empty text")

            try:
                with api_limiter.limited_context("openai", "embeddings"):
                    logger.debug(f"Generating embedding for text: {text[:50]}...")

                    response = self.client.embeddings.create(
                        input=text, model=self.model
                    )

                    embedding = response.data[0].embedding
                    logger.debug(
                        f"Successfully generated embedding with {len(embedding)} dimensions"
                    )

                    return embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise APIException(
                    message=f"Failed to generate embedding: {str(e)}",
                    service="OpenAI Embeddings",
                )

    def get_embeddings_batch(
        self, texts: List[str], batch_size: int = 10
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts using batched API calls.

        Args:
            texts (list): List of texts to embed
            batch_size (int): Number of texts per API call

        Returns:
            list: List of embedding vectors
        """
        with ErrorHandler(error_type="batch_embedding", reraise=True):
            if not texts:
                logger.warning("Attempted to embed empty text batch")
                return []

            # Filter out empty texts
            filtered_texts = [text for text in texts if text and not text.isspace()]

            if not filtered_texts:
                logger.warning("All texts in batch were empty")
                return []

            all_embeddings = []

            # Process in batches
            for i in range(0, len(filtered_texts), batch_size):
                batch = filtered_texts[i : i + batch_size]

                try:
                    with api_limiter.limited_context("openai", "embeddings"):
                        logger.debug(
                            f"Generating embeddings for batch of {len(batch)} texts"
                        )

                        response = self.client.embeddings.create(
                            input=batch, model=self.model
                        )

                        # Sort embeddings by index to maintain original order
                        sorted_data = sorted(response.data, key=lambda x: x.index)
                        batch_embeddings = [item.embedding for item in sorted_data]

                        all_embeddings.extend(batch_embeddings)

                        logger.debug(
                            f"Successfully generated {len(batch_embeddings)} embeddings"
                        )

                        # Small delay between batches to avoid rate limiting
                        if i + batch_size < len(filtered_texts):
                            time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error generating batch embeddings: {str(e)}")
                    raise APIException(
                        message=f"Failed to generate batch embeddings: {str(e)}",
                        service="OpenAI Embeddings",
                    )

            # If some texts were empty, fill with None to match original length
            if len(all_embeddings) < len(texts):
                result = []
                embedding_idx = 0

                for text in texts:
                    if text and not text.isspace():
                        result.append(all_embeddings[embedding_idx])
                        embedding_idx += 1
                    else:
                        # For empty text, add a zero vector with the correct dimension
                        # Use the dimension of the first valid embedding
                        if all_embeddings:
                            result.append([0.0] * len(all_embeddings[0]))
                        else:
                            # Default OpenAI embedding dimension if no valid embeddings
                            result.append([0.0] * 1536)

                return result

            return all_embeddings

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1 (list): First embedding vector
            embedding2 (list): Second embedding vector

        Returns:
            float: Cosine similarity score (0-1)
        """
        if not embedding1 or not embedding2:
            return 0.0

        if len(embedding1) != len(embedding2):
            logger.warning(
                f"Embedding dimensions don't match: {len(embedding1)} vs {len(embedding2)}"
            )
            return 0.0

        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

            # Calculate magnitudes
            magnitude1 = sum(a * a for a in embedding1) ** 0.5
            magnitude2 = sum(b * b for b in embedding2) ** 0.5

            # Calculate cosine similarity
            if magnitude1 > 0 and magnitude2 > 0:
                similarity = dot_product / (magnitude1 * magnitude2)
                return similarity
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
