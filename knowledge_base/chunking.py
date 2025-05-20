"""
Text chunking strategies for the Manzil Chatbot knowledge base.
"""

import re
from typing import List, Dict, Any
from utils.logging_config import get_logger
from utils.error_handlers import ErrorHandler
from config import settings

logger = get_logger("knowledge_base.chunking")


class TextChunker:
    """
    Handles the chunking of text content for vector embedding and storage.
    """

    def __init__(self):
        """Initialize the text chunker with configuration settings."""
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.min_chunk_length = settings.MIN_CHUNK_LENGTH

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces for embedding.

        Args:
            document (dict): The document to chunk, containing 'content' and metadata

        Returns:
            list: List of chunk dictionaries with content and metadata
        """
        with ErrorHandler(error_type="text_chunking", reraise=True):
            if not document or "content" not in document:
                logger.warning("Empty document or missing content field")
                return []

            content = document["content"]

            # Create chunks
            chunks = self._create_semantic_chunks(content)

            # Create result with metadata
            result = []
            for i, chunk in enumerate(chunks):
                # Skip very short chunks
                if len(chunk.split()) < self.min_chunk_length:
                    continue

                chunk_dict = {
                    "content": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }

                # Copy metadata from original document
                for key, value in document.items():
                    if key != "content" and key != "structured_content":
                        chunk_dict[key] = value

                # Add document title to each chunk
                if "title" in document:
                    chunk_dict["document_title"] = document["title"]

                result.append(chunk_dict)

            logger.debug(
                f"Created {len(result)} chunks from document '{document.get('title', 'Untitled')}'"
            )
            return result

    def _create_semantic_chunks(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks with overlap.

        This method tries to create chunks that respect semantic boundaries like
        paragraphs, sentences, and even header sections.

        Args:
            text (str): The text to chunk

        Returns:
            list: List of text chunks
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Try to split by paragraphs first
        paragraphs = self._split_by_paragraphs(text)

        # If we have very large paragraphs, split them into sentences
        paragraphs = self._split_large_paragraphs(paragraphs)

        # Now combine paragraphs into chunks with overlap
        chunks = self._combine_with_overlap(paragraphs)

        return chunks

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by blank lines (two or more newlines)
        raw_paragraphs = re.split(r"\n\s*\n", text)

        # Clean up and filter empty paragraphs
        return [p.strip() for p in raw_paragraphs if p.strip()]

    def _split_large_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Split large paragraphs into sentences."""
        result = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                result.append(paragraph)
            else:
                # Split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                result.extend(sentences)

        return result

    def _combine_with_overlap(self, segments: List[str]) -> List[str]:
        """Combine segments into chunks with overlap."""
        chunks = []
        current_chunk = ""

        for segment in segments:
            # If adding this segment would exceed chunk size and
            # we already have some content, store the current chunk
            if (
                current_chunk
                and len(current_chunk) + len(segment) + 1 > self.chunk_size
            ):
                chunks.append(current_chunk)

                # Start a new chunk with overlap from the previous chunk
                if len(current_chunk) > self.chunk_overlap:
                    # Try to find a sentence boundary for the overlap
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    sentence_break = max(
                        overlap_text.rfind(". "),
                        overlap_text.rfind("! "),
                        overlap_text.rfind("? "),
                    )

                    if sentence_break > 0:
                        # Start from the beginning of the last sentence in the overlap
                        current_chunk = current_chunk[
                            -(self.chunk_overlap - sentence_break) :
                        ]
                    else:
                        # No sentence break found, use word boundaries
                        words = current_chunk.split()
                        if len(words) > 5:  # Ensure we have enough words
                            # Start from the last ~5 words
                            current_chunk = " ".join(words[-5:])
                        else:
                            current_chunk = ""
                else:
                    # If chunk is smaller than overlap, just use the whole chunk
                    current_chunk = current_chunk

            # Add segment to the current chunk
            if current_chunk:
                current_chunk += " " + segment
            else:
                current_chunk = segment

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_by_headers(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document by headers for more semantic division.

        This is especially useful for help center pages with sections.

        Args:
            document (dict): Document with 'content' and 'structured_content'

        Returns:
            list: List of chunk dictionaries with content and metadata
        """
        with ErrorHandler(
            error_type="header_chunking", reraise=False, fallback_value=[]
        ):
            if (
                "structured_content" not in document
                or "headers" not in document["structured_content"]
            ):
                # Fall back to regular chunking
                return self.chunk_document(document)

            headers = document["structured_content"]["headers"]
            content = document["content"]

            if not headers or len(headers) <= 1:
                # Not enough headers to chunk by, use regular chunking
                return self.chunk_document(document)

            # Sort headers by their position in the text
            headers.sort(key=lambda h: content.find(h["text"]))

            # Create chunks based on headers
            chunks = []
            for i, header in enumerate(headers):
                # Find the start of this section
                section_start = content.find(header["text"])

                if section_start == -1:
                    continue

                # Find the end of this section (start of next section or end of document)
                if i < len(headers) - 1:
                    next_header_start = content.find(headers[i + 1]["text"])
                    if next_header_start > section_start:
                        section_text = content[section_start:next_header_start]
                    else:
                        # Headers might be out of order in the text
                        section_text = content[section_start:]
                else:
                    section_text = content[section_start:]

                # Create chunk for this section
                chunk_dict = {
                    "content": section_text.strip(),
                    "section_title": header["text"],
                    "section_level": header["level"],
                    "chunk_index": i,
                    "total_chunks": len(headers),
                }

                # Copy metadata from original document
                for key, value in document.items():
                    if key != "content" and key != "structured_content":
                        chunk_dict[key] = value

                # Add document title to each chunk
                if "title" in document:
                    chunk_dict["document_title"] = document["title"]

                chunks.append(chunk_dict)

            # If any of the resulting chunks are too large, apply normal chunking
            final_chunks = []
            for chunk in chunks:
                if len(chunk["content"]) > self.chunk_size:
                    # Create a temporary document for chunking
                    temp_doc = {
                        "content": chunk["content"],
                        "title": chunk.get("section_title", ""),
                    }
                    # Add other metadata
                    for key, value in chunk.items():
                        if key != "content" and key != "title":
                            temp_doc[key] = value

                    # Chunk this section
                    section_chunks = self.chunk_document(temp_doc)
                    final_chunks.extend(section_chunks)
                else:
                    final_chunks.append(chunk)

            logger.debug(
                f"Created {len(final_chunks)} chunks by headers from document '{document.get('title', 'Untitled')}'"
            )
            return final_chunks

    def get_chunk_with_context(
        self, chunks: List[Dict[str, Any]], chunk_index: int
    ) -> str:
        """
        Get a chunk with surrounding context from other chunks.

        Args:
            chunks (list): List of chunk dictionaries
            chunk_index (int): Index of the chunk to get context for

        Returns:
            str: The chunk content with surrounding context
        """
        if not chunks:
            return ""

        # Validate chunk_index
        if chunk_index < 0 or chunk_index >= len(chunks):
            return chunks[0]["content"] if chunks else ""

        # Get the main chunk
        main_chunk = chunks[chunk_index]["content"]

        # Get previous and next chunks for context
        prev_chunk = chunks[chunk_index - 1]["content"] if chunk_index > 0 else ""
        next_chunk = (
            chunks[chunk_index + 1]["content"] if chunk_index < len(chunks) - 1 else ""
        )

        # Combine with context
        result = ""
        if prev_chunk:
            result += f"Previous context: {prev_chunk}\n\n"

        result += f"Main content: {main_chunk}\n\n"

        if next_chunk:
            result += f"Following context: {next_chunk}"

        return result
