"""
Vector database management for the Manzil Chatbot knowledge base.
"""

import time
import json
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings

from config import settings
from utils.logging_config import get_logger
from utils.error_handlers import VectorStoreException, ErrorHandler, retry
from nlp.embeddings import EmbeddingGenerator

logger = get_logger("knowledge_base.vector_store")


class VectorStore:
    """
    Manages the vector database for storing and retrieving embeddings.
    """

    def __init__(self):
        """Initialize the vector store with configuration settings."""
        self.collection_name = settings.COLLECTION_NAME
        self.persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()

        # Initialize ChromaDB client and collection
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Persistent client with directory storage
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(
                    f"Using existing ChromaDB collection: {self.collection_name}"
                )
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Manzil help center content embeddings"},
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise VectorStoreException(
                f"Vector database initialization failed: {str(e)}"
            )

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            peek = self.collection.peek(10)

            return {
                "name": self.collection_name,
                "count": count,
                "sample": peek,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {"name": self.collection_name, "error": str(e)}

    def clear_collection(self):
        """Clear all documents from the collection."""
        with ErrorHandler(error_type="vector_store", reraise=True):
            try:
                self.collection.delete(where={})
                logger.info(f"Cleared collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to clear collection: {str(e)}")
                raise VectorStoreException(f"Failed to clear vector database: {str(e)}")

    def _prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare metadata for ChromaDB storage, ensuring all values are strings.

        Args:
            chunk (dict): The chunk dictionary with metadata

        Returns:
            dict: Metadata dictionary with string values
        """
        metadata = {}

        # Copy all fields except 'content' to metadata
        for key, value in chunk.items():
            if key != "content":
                # Convert value to string if not already
                if isinstance(value, (dict, list)):
                    metadata[key] = json.dumps(value)
                else:
                    metadata[key] = str(value)

        return metadata

    def _generate_document_id(self, chunk: Dict[str, Any]) -> str:
        """
        Generate a unique document ID for a chunk.

        Args:
            chunk (dict): The chunk dictionary

        Returns:
            str: A unique document ID
        """
        # Base the ID on URL and chunk index if available
        url = chunk.get("url", "")
        chunk_index = chunk.get("chunk_index", 0)

        if url:
            # Clean URL to use as part of ID
            clean_url = (
                url.replace("https://", "").replace("http://", "").replace("/", "_")
            )
            return f"chunk_{clean_url}_{chunk_index}"
        else:
            # Fallback to timestamp-based ID
            timestamp = chunk.get("crawl_time", time.time())
            return f"chunk_{int(timestamp)}_{chunk_index}"

    @retry(exceptions=(Exception,), tries=3, delay=2, backoff=2, logger=logger)
    def store_documents(self, chunks: List[Dict[str, Any]]):
        """
        Store document chunks in the vector database.

        Args:
            chunks (list): List of chunk dictionaries with content and metadata
        """
        with ErrorHandler(error_type="vector_store", reraise=True):
            if not chunks:
                logger.warning("No chunks to store")
                return

            logger.info(f"Storing {len(chunks)} chunks in vector database")

            try:
                # Process chunks in batches to avoid memory issues
                for i in range(0, len(chunks), self.batch_size):
                    batch = chunks[i : i + self.batch_size]
                    self._store_batch(batch)
                    logger.debug(f"Stored batch of {len(batch)} chunks")

                logger.info(
                    f"Successfully stored {len(chunks)} chunks in vector database"
                )
            except Exception as e:
                logger.error(f"Failed to store documents: {str(e)}")
                raise VectorStoreException(
                    f"Failed to store documents in vector database: {str(e)}"
                )

    def _store_batch(self, chunks: List[Dict[str, Any]]):
        """
        Store a batch of chunks in the vector database.

        Args:
            chunks (list): List of chunk dictionaries to store
        """
        documents = []
        metadatas = []
        ids = []

        # Generate embeddings for all documents in batch
        contents = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_generator.get_embeddings_batch(contents)

        if len(embeddings) != len(chunks):
            raise VectorStoreException(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(chunks)}"
            )

        # Prepare data for ChromaDB
        for i, chunk in enumerate(chunks):
            documents.append(chunk["content"])
            metadatas.append(self._prepare_metadata(chunk))
            ids.append(self._generate_document_id(chunk))

        # Add to collection
        self.collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents to a query.

        Args:
            query (str): The query text
            top_k (int): Number of results to return

        Returns:
            list: List of similar documents with metadata
        """
        with ErrorHandler(error_type="vector_store_query", reraise=True):
            try:
                # Generate embedding for query
                query_embedding = self.embedding_generator.get_embedding(query)

                # Query the collection
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )

                # Format results
                formatted_results = []

                if not results["documents"] or not results["documents"][0]:
                    logger.warning(f"No results found for query: {query}")
                    return []

                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]

                    # Convert JSON string fields back to objects
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        if (
                            isinstance(value, str)
                            and value.startswith("{")
                            and value.endswith("}")
                        ):
                            try:
                                parsed_metadata[key] = json.loads(value)
                            except json.JSONDecodeError:
                                parsed_metadata[key] = value
                        else:
                            parsed_metadata[key] = value

                    formatted_results.append(
                        {
                            "content": doc,
                            "metadata": parsed_metadata,
                            "distance": (
                                results["distances"][0][i]
                                if "distances" in results
                                else None
                            ),
                        }
                    )

                logger.debug(
                    f"Retrieved {len(formatted_results)} results for query: {query[:50]}..."
                )
                return formatted_results
            except Exception as e:
                logger.error(f"Failed to retrieve similar documents: {str(e)}")
                raise VectorStoreException(
                    f"Failed to retrieve similar documents: {str(e)}"
                )

    def search_by_metadata(
        self, metadata_filter: Dict[str, Any], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata.

        Args:
            metadata_filter (dict): Filter criteria for metadata
            limit (int): Maximum number of results to return

        Returns:
            list: List of matching documents with metadata
        """
        with ErrorHandler(error_type="vector_store_search", reraise=True):
            try:
                # Convert filter to ChromaDB where clause
                where_clause = {}
                for key, value in metadata_filter.items():
                    where_clause[key] = str(value)

                # Query the collection
                results = self.collection.get(
                    where=where_clause, limit=limit, include=["documents", "metadatas"]
                )

                # Format results
                formatted_results = []

                if not results["documents"]:
                    logger.warning(
                        f"No results found for metadata filter: {metadata_filter}"
                    )
                    return []

                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i]

                    # Convert JSON string fields back to objects
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        if (
                            isinstance(value, str)
                            and value.startswith("{")
                            and value.endswith("}")
                        ):
                            try:
                                parsed_metadata[key] = json.loads(value)
                            except json.JSONDecodeError:
                                parsed_metadata[key] = value
                        else:
                            parsed_metadata[key] = value

                    formatted_results.append(
                        {"content": doc, "metadata": parsed_metadata}
                    )

                logger.debug(
                    f"Found {len(formatted_results)} results for metadata filter: {metadata_filter}"
                )
                return formatted_results
            except Exception as e:
                logger.error(f"Failed to search by metadata: {str(e)}")
                raise VectorStoreException(f"Failed to search by metadata: {str(e)}")
