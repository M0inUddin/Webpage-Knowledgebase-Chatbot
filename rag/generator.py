"""
Response generation using OpenAI's LLM for the Manzil Chatbot.
"""

from typing import Dict, List, Any, Optional
from openai import OpenAI
from ratelimiter import RateLimiter

from config import settings
from utils.logging_config import get_logger
from utils.error_handlers import RAGException, retry, ErrorHandler, APIException

logger = get_logger("rag.generator")


class ResponseGenerator:
    """
    Generates responses using OpenAI's language models.
    """

    def __init__(self):
        """Initialize the response generator with API settings."""
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.COMPLETION_MODEL
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        self.rate_limit = settings.RATE_LIMIT

        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized response generator with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise RAGException(f"Failed to initialize response generator: {str(e)}")

        # Set up rate limiter
        self.rate_limiter = RateLimiter(max_calls=self.rate_limit, period=60)

    @retry(exceptions=(Exception,), tries=3, delay=2, backoff=2, logger=logger)
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using OpenAI's language model.

        Args:
            prompt (str): The prompt for the LLM
            temperature (float, optional): Temperature parameter (0.0-1.0)
            max_tokens (int, optional): Maximum tokens in the response

        Returns:
            str: Generated response
        """
        with ErrorHandler(error_type="response_generation", reraise=True):
            if not prompt:
                logger.warning("Attempted to generate with empty prompt")
                raise ValueError("Cannot generate response from empty prompt")

            try:
                # Use provided parameters or defaults
                temp = temperature if temperature is not None else self.temperature
                max_tok = max_tokens if max_tokens is not None else self.max_tokens

                with self.rate_limiter:
                    logger.debug(
                        f"Generating response with prompt of length: {len(prompt)}"
                    )

                    # Split prompt into system and user message
                    system_message, user_message = self._parse_prompt(prompt)

                    messages = [{"role": "system", "content": system_message}]

                    # Add user message if split was successful
                    if user_message:
                        messages.append({"role": "user", "content": user_message})

                    # Generate response
                    response = self.client.responses.create(
                        model=self.model,
                        input=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                    )

                    generated_text = response.choices[0].message.content
                    logger.debug(f"Generated response of length: {len(generated_text)}")

                    return generated_text
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise APIException(
                    message=f"Failed to generate response: {str(e)}",
                    service="OpenAI Completions",
                )

    def _parse_prompt(self, prompt: str) -> tuple:
        """
        Parse a combined prompt into system and user messages.

        Args:
            prompt (str): Combined prompt string

        Returns:
            tuple: (system_message, user_message)
        """
        # Look for "User:" to split the prompt
        parts = prompt.split("\n\nUser: ", 1)

        if len(parts) == 2:
            system_message = parts[0]
            user_message = parts[1]
        else:
            # If no clear user message, use the whole prompt as system message
            system_message = prompt
            user_message = ""

        return system_message, user_message

    def generate_with_streaming(
        self,
        prompt: str,
        callback,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate a response with streaming for real-time updates.

        Args:
            prompt (str): The prompt for the LLM
            callback (callable): Function to call with each chunk of generated text
            temperature (float, optional): Temperature parameter (0.0-1.0)
            max_tokens (int, optional): Maximum tokens in the response
        """
        with ErrorHandler(error_type="streaming_generation", reraise=True):
            if not prompt:
                logger.warning("Attempted to generate with empty prompt")
                raise ValueError("Cannot generate response from empty prompt")

            try:
                # Use provided parameters or defaults
                temp = temperature if temperature is not None else self.temperature
                max_tok = max_tokens if max_tokens is not None else self.max_tokens

                with self.rate_limiter:
                    logger.debug(
                        f"Generating streaming response with prompt of length: {len(prompt)}"
                    )

                    # Split prompt into system and user message
                    system_message, user_message = self._parse_prompt(prompt)

                    messages = [{"role": "system", "content": system_message}]

                    # Add user message if split was successful
                    if user_message:
                        messages.append({"role": "user", "content": user_message})

                    # Generate streaming response
                    stream = self.client.responses.create(
                        model=self.model,
                        input=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                        stream=True,
                    )

                    # Process the stream
                    full_response = ""
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            callback(content)

                    logger.debug(
                        f"Completed streaming response of length: {len(full_response)}"
                    )
                    return full_response
            except Exception as e:
                logger.error(f"Error generating streaming response: {str(e)}")
                raise APIException(
                    message=f"Failed to generate streaming response: {str(e)}",
                    service="OpenAI Completions",
                )

    def generate_islamic_finance_explanation(self, term: str, context: str) -> str:
        """
        Generate an explanation of an Islamic finance term.

        Args:
            term (str): The term to explain
            context (str): Context information about the term

        Returns:
            str: Generated explanation
        """
        prompt = f"""You are a helpful assistant for {settings.COMPANY_NAME}, specializing in Islamic finance.

Explain the Islamic finance term "{term}" based ONLY on the provided context. 

Include:
1. A clear definition of the term
2. How it differs from conventional finance alternatives (if mentioned in the context)
3. Its application in {settings.COMPANY_NAME}'s products (if mentioned in the context)
4. Any relevant Shariah principles associated with it (if mentioned in the context)

Context:
{context}

Format your response in a structured, educational manner suitable for someone unfamiliar with Islamic finance.
"""

        return self.generate(prompt, temperature=0.3)

    def generate_concise_answer(self, prompt: str) -> str:
        """
        Generate a more concise answer for mobile users or quick responses.

        Args:
            prompt (str): The prompt for the LLM

        Returns:
            str: Concise generated response
        """
        # Modify the prompt to request a more concise response
        concise_prompt = (
            prompt
            + "\n\nPlease provide a concise response suitable for mobile users, focusing on the most essential information."
        )

        return self.generate(concise_prompt, max_tokens=150)

    def generate_source_citations(
        self, prompt: str, sources: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a response with explicit source citations.

        Args:
            prompt (str): The prompt for the LLM
            sources (list): List of source documents

        Returns:
            str: Generated response with citations
        """
        # Add source citation instructions
        citation_prompt = (
            prompt
            + "\n\nImportant: For each key piece of information in your response, cite the specific document number that contains it using the format [Doc X]. At the end of your response, list all the sources you referenced."
        )

        # Add source details
        source_details = "\n\nSource details:\n"
        for i, source in enumerate(sources):
            title = source.get("metadata", {}).get("title", "Untitled")
            url = source.get("metadata", {}).get("url", "No URL")
            source_details += f"[Doc {i+1}] {title} - {url}\n"

        full_prompt = citation_prompt + source_details

        return self.generate(full_prompt)
