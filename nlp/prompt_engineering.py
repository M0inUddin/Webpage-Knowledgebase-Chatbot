"""
Prompt engineering strategies for the Manzil Chatbot.
"""

from typing import List, Dict, Any, Optional
from config import settings
from utils.logging_config import get_logger

logger = get_logger("nlp.prompt_engineering")


class PromptEngineering:
    """
    Creates and manages prompts for the RAG system.
    """

    def __init__(self):
        """Initialize prompt engineering with domain-specific information."""
        self.company_name = settings.COMPANY_NAME
        self.company_description = settings.COMPANY_DESCRIPTION
        self.services = settings.SERVICES

    def create_system_prompt(self, query_type: str) -> str:
        """
        Create a system prompt based on query type.

        Args:
            query_type (str): Type of query (factual, exploratory, etc.)

        Returns:
            str: Appropriate system prompt
        """
        base_prompt = f"You are a helpful assistant for {self.company_name}, {self.company_description}"

        if query_type == "factual":
            return f"{base_prompt}\n\nThe user is asking for factual information. Answer the question based ONLY on the provided context. Be direct and precise. Cite the source document number in your answer."

        elif query_type == "exploratory":
            return f"{base_prompt}\n\nThe user is exploring options or possibilities. Provide a comprehensive overview of relevant information from the context. Organize your answer by categories or options if appropriate. Remember to only use information from the provided context."

        elif query_type == "process":
            return f"{base_prompt}\n\nThe user is asking about a process or procedure. Provide step-by-step instructions based ONLY on the provided context. Present the steps in a clear, sequential order. If the context doesn't contain complete process information, acknowledge the limitations."

        elif query_type == "comparison":
            return f"{base_prompt}\n\nThe user is asking for a comparison. Compare the options based ONLY on the provided context. Present the comparison in a structured format, highlighting key differences and similarities. Only include information present in the context."

        elif query_type == "islamic_finance":
            return f"{base_prompt}\n\nThe user is asking about Islamic finance principles or Shariah-compliant products. Explain the concepts based ONLY on the provided context. Be accurate about Islamic finance terminology and principles. If the context doesn't contain complete information, acknowledge the limitations."

        elif query_type == "out_of_scope":
            return f"{base_prompt}\n\nIf the user's question cannot be answered based on the provided context, clearly state that you don't have that specific information and suggest contacting {self.company_name} directly. Do not attempt to answer with information not in the context."

        else:  # general
            return f"{base_prompt}\n\nYour purpose is to provide accurate information based ONLY on the context provided below. If the information needed to answer the question is not in the context, politely indicate this and suggest contacting {self.company_name} directly. Do not make up information or draw from knowledge outside the provided context."

    def format_context(self, retrieved_documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            retrieved_documents (list): List of retrieved document dictionaries

        Returns:
            str: Formatted context string
        """
        if not retrieved_documents:
            return "No relevant information found."

        context_parts = []

        for i, doc in enumerate(retrieved_documents):
            content = doc["content"]
            metadata = doc.get("metadata", {})

            # Get document title and URL
            title = metadata.get(
                "document_title", metadata.get("title", "Untitled Document")
            )
            url = metadata.get("url", "No source URL")

            # Get section title if available
            section = metadata.get("section_title", "")
            section_prefix = f" - {section}" if section else ""

            # Format document with header
            document_header = f"Document {i+1}: {title}{section_prefix}"
            document_content = f"{document_header}\n{content}\nSource: {url}\n"

            context_parts.append(document_content)

        return "\n\n".join(context_parts)

    def create_rag_prompt(
        self,
        query_type: str,
        user_query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Create a full RAG prompt including system message, context, and user query.

        Args:
            query_type (str): Type of query
            user_query (str): The user's question
            context (str): Formatted context from retrieved documents
            conversation_history (list, optional): Previous conversation turns

        Returns:
            str: Complete prompt for the LLM
        """
        system_prompt = self.create_system_prompt(query_type)

        # Add conversation history summary if available
        history_summary = ""
        if conversation_history and len(conversation_history) > 0:
            history_summary = "Previous conversation summary:\n"
            for turn in conversation_history[-3:]:  # Include last 3 turns at most
                if "user" in turn:
                    history_summary += f"User: {turn['user']}\n"
                if "assistant" in turn:
                    history_summary += f"Assistant: {turn['assistant']}\n"
            history_summary += "\n"

        # Include services list for general orientation
        services_info = (
            "Our services include:\n- " + "\n- ".join(self.services) + "\n\n"
        )

        # Assemble final prompt
        prompt = f"{system_prompt}\n\n"

        if history_summary:
            prompt += f"{history_summary}\n"

        prompt += f"{services_info}Context:\n{context}\n\nUser: {user_query}"

        return prompt

    def create_clarification_prompt(self, user_query: str, ambiguity_type: str) -> str:
        """
        Create a prompt to clarify ambiguous queries.

        Args:
            user_query (str): The user's ambiguous question
            ambiguity_type (str): Type of ambiguity (e.g., "product", "service", "term")

        Returns:
            str: Prompt for generating clarification question
        """
        base_prompt = f"You are a helpful assistant for {self.company_name}. The user has asked an ambiguous question."

        if ambiguity_type == "product":
            return f"{base_prompt}\n\nThe user's question about products is ambiguous. Generate a response that asks for clarification about which specific product they're interested in. List the relevant products from {self.company_name} that might address their query.\n\nUser query: {user_query}"

        elif ambiguity_type == "service":
            return f"{base_prompt}\n\nThe user's question about services is ambiguous. Generate a response that asks for clarification about which specific service they're interested in. List the relevant services from {self.company_name} that might address their query.\n\nUser query: {user_query}"

        elif ambiguity_type == "term":
            return f"{base_prompt}\n\nThe user's question uses an ambiguous financial or Islamic finance term. Generate a response that asks for clarification about what they mean by this term, and offer a few possible interpretations.\n\nUser query: {user_query}"

        else:  # general ambiguity
            return f"{base_prompt}\n\nGenerate a response that politely asks for clarification about their question, suggesting a few specific aspects they might want to elaborate on.\n\nUser query: {user_query}"

    def create_no_context_prompt(self, user_query: str) -> str:
        """
        Create a prompt for when no relevant context is found.

        Args:
            user_query (str): The user's question

        Returns:
            str: Prompt for generating a response without context
        """
        prompt = f"""You are a helpful assistant for {self.company_name}, {self.company_description}

The user has asked a question, but we couldn't find specific information in our knowledge base to answer it accurately.

Generate a polite response that:
1. Acknowledges that you don't have the specific information they're looking for
2. Suggests contacting {self.company_name} directly for the most accurate information
3. If appropriate, mentions the general services {self.company_name} offers that might be relevant to their query
4. Provides contact information or suggests visiting the website for more details

User query: {user_query}

Available services:
- {"- ".join(self.services)}
"""
        return prompt

    def create_islamic_finance_explanation_prompt(self, term: str, context: str) -> str:
        """
        Create a prompt for explaining Islamic finance terms.

        Args:
            term (str): The Islamic finance term to explain
            context (str): Any available context about the term

        Returns:
            str: Prompt for generating an explanation
        """
        prompt = f"""You are a helpful assistant for {self.company_name}, specializing in Islamic finance.

Explain the Islamic finance term "{term}" based ONLY on the provided context. 

Include:
1. A clear definition of the term
2. How it differs from conventional finance alternatives (if mentioned in the context)
3. Its application in {self.company_name}'s products (if mentioned in the context)
4. Any relevant Shariah principles associated with it (if mentioned in the context)

Context:
{context}

Format your response in a structured, educational manner suitable for someone unfamiliar with Islamic finance.
"""
        return prompt
