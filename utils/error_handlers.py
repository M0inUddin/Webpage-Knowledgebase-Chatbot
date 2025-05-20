"""
Custom exception classes and error handling utilities for the Manzil Chatbot.
"""

import traceback
import time
import functools
import logging

logger = logging.getLogger("utils.error_handlers")


# Custom exception classes
class ManzilChatbotException(Exception):
    """Base exception class for all Manzil Chatbot exceptions."""

    pass


class CrawlerException(ManzilChatbotException):
    """Exception raised for errors in the web crawler."""

    pass


class ContentExtractionException(CrawlerException):
    """Exception raised for errors during content extraction."""

    pass


class VectorStoreException(ManzilChatbotException):
    """Exception raised for vector database errors."""

    pass


class EmbeddingException(ManzilChatbotException):
    """Exception raised for errors in the embedding process."""

    pass


class RAGException(ManzilChatbotException):
    """Exception raised for errors in the RAG pipeline."""

    pass


class APIException(ManzilChatbotException):
    """Exception raised for errors in API calls."""

    def __init__(self, message, status_code=None, service=None):
        self.status_code = status_code
        self.service = service
        super().__init__(message)


# Decorator for retry logic
def retry(exceptions, tries=4, delay=3, backoff=2, logger=None):
    """
    Retry decorator with exponential backoff.

    Args:
        exceptions: The exceptions to catch and retry.
        tries: Number of times to try (not retry) before giving up.
        delay: Initial delay between retries in seconds.
        backoff: Backoff multiplier e.g. value of 2 will double the delay each retry.
        logger: Logger to use. If None, print.
    """

    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            last_exception = None

            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    msg = f"Retrying {f.__name__} in {mdelay} seconds... ({mtries-1} tries remaining)"

                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)

                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff

            try:
                return f(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                msg = f"Function {f.__name__} failed after {tries} tries"

                if logger:
                    logger.error(msg)
                else:
                    print(msg)

                raise last_exception

        return f_retry

    return deco_retry


# Error handling context manager
class ErrorHandler:
    """Context manager for error handling with specific strategies."""

    def __init__(
        self, error_type="general", fallback_value=None, log_level="error", reraise=True
    ):
        """
        Initialize the error handler.

        Args:
            error_type (str): Type of error for logging purposes.
            fallback_value: Value to return if an error occurs and reraise is False.
            log_level (str): Logging level to use ('error', 'warning', 'critical', 'info').
            reraise (bool): Whether to re-raise the exception after handling.
        """
        self.error_type = error_type
        self.fallback_value = fallback_value
        self.log_level = getattr(logging, log_level.upper(), logging.ERROR)
        self.reraise = reraise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log the error
            error_message = f"{self.error_type.capitalize()} error: {str(exc_val)}"
            error_traceback = traceback.format_exception(exc_type, exc_val, exc_tb)

            if self.log_level == logging.ERROR:
                logger.error(error_message)
                logger.debug("".join(error_traceback))
            elif self.log_level == logging.WARNING:
                logger.warning(error_message)
            elif self.log_level == logging.CRITICAL:
                logger.critical(error_message)
                logger.debug("".join(error_traceback))
            else:
                logger.info(error_message)

            # Don't re-raise if we have a fallback
            if not self.reraise:
                return True  # Suppress the exception

        # Let the exception propagate
        return False


# Function to format exception information for user-friendly display
def format_user_error(exception, include_trace=False):
    """
    Format an exception for user-friendly display.

    Args:
        exception: The exception to format.
        include_trace (bool): Whether to include the traceback.

    Returns:
        str: A user-friendly error message.
    """
    error_type = exception.__class__.__name__
    error_message = str(exception)

    if isinstance(exception, APIException):
        # API-specific formatting
        service = exception.service or "external service"
        status = f" (Status: {exception.status_code})" if exception.status_code else ""
        return f"Error communicating with {service}{status}. {error_message}"

    elif isinstance(exception, CrawlerException):
        return f"Error retrieving content: {error_message}"

    elif isinstance(exception, EmbeddingException):
        return (
            "Error processing your query. Please try again with a different question."
        )

    elif isinstance(exception, RAGException):
        return f"Error generating response: {error_message}"

    else:
        # Generic formatting
        if include_trace and isinstance(exception, ManzilChatbotException):
            trace = traceback.format_exc()
            return f"An error occurred: {error_message}\n\nDetails: {trace}"
        else:
            return f"An error occurred: {error_message}"
