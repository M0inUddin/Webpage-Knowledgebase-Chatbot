"""
Rate limiting utilities for the Manzil Chatbot.
"""

import time
import threading
from typing import Dict, Optional
from functools import wraps

from config import settings
from utils.logging_config import get_logger

logger = get_logger("utils.rate_limiters")


class TokenBucket:
    """
    Token bucket rate limiter implementation.

    This class implements the token bucket algorithm for rate limiting,
    which allows for bursts of requests up to a certain limit while
    maintaining a long-term rate limit.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize the token bucket.

        Args:
            rate (float): Rate at which tokens are added (tokens per second)
            capacity (int): Maximum number of tokens the bucket can hold
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.RLock()

    def _add_tokens(self):
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.rate

        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens (int): Number of tokens to consume

        Returns:
            bool: Whether the tokens were successfully consumed
        """
        with self.lock:
            self._add_tokens()

            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            else:
                return False

    def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait until the requested number of tokens is available.

        Args:
            tokens (int): Number of tokens to wait for
            timeout (float, optional): Maximum time to wait in seconds

        Returns:
            bool: Whether the tokens were successfully consumed
        """
        start_time = time.time()

        # First try without waiting
        if self.consume(tokens):
            return True

        # Calculate time to wait for enough tokens
        with self.lock:
            self._add_tokens()

            if tokens > self.capacity:
                logger.warning(
                    f"Requested tokens ({tokens}) exceed bucket capacity ({self.capacity})"
                )
                return False

            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate

            if timeout is not None and wait_time > timeout:
                logger.warning(
                    f"Wait time ({wait_time:.2f}s) would exceed timeout ({timeout}s)"
                )
                return False

        # Wait and try again
        if wait_time > 0:
            time.sleep(wait_time)

        return self.consume(tokens)


class RateLimiter:
    """
    Rate limiter for API calls and other operations.

    This class provides rate limiting functionality with per-key limits,
    allowing for different limits for different resources or operations.
    """

    def __init__(self, default_rate: float = None, default_capacity: int = None):
        """
        Initialize the rate limiter.

        Args:
            default_rate (float, optional): Default rate (tokens per second)
            default_capacity (int, optional): Default bucket capacity
        """
        self.default_rate = (
            default_rate or settings.RATE_LIMIT / 60.0
        )  # Default to settings or 1 per minute
        self.default_capacity = default_capacity or min(
            10, self.default_rate * 10
        )  # Default capacity
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.RLock()

    def get_bucket(
        self, key: str, rate: Optional[float] = None, capacity: Optional[int] = None
    ) -> TokenBucket:
        """
        Get or create a token bucket for the specified key.

        Args:
            key (str): The resource or operation key
            rate (float, optional): Rate for this specific bucket
            capacity (int, optional): Capacity for this specific bucket

        Returns:
            TokenBucket: The token bucket instance
        """
        with self.lock:
            if key not in self.buckets:
                bucket_rate = rate or self.default_rate
                bucket_capacity = capacity or self.default_capacity
                self.buckets[key] = TokenBucket(bucket_rate, bucket_capacity)
            return self.buckets[key]

    def limit(self, key: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Apply rate limiting to an operation.

        Args:
            key (str): The resource or operation key
            tokens (int): Number of tokens to consume
            timeout (float, optional): Maximum time to wait in seconds

        Returns:
            bool: Whether the operation should proceed
        """
        bucket = self.get_bucket(key)
        return bucket.wait_for_tokens(tokens, timeout)

    def wait(self, key: str, tokens: int = 1):
        """
        Wait until the rate limit allows an operation to proceed.

        Args:
            key (str): The resource or operation key
            tokens (int): Number of tokens to consume
        """
        bucket = self.get_bucket(key)

        # First try without waiting
        if bucket.consume(tokens):
            return

        # Calculate wait time
        with bucket.lock:
            bucket._add_tokens()
            tokens_needed = tokens - bucket.tokens
            wait_time = tokens_needed / bucket.rate

        # Wait and then consume
        if wait_time > 0:
            logger.debug(f"Rate limit hit for {key}, waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        bucket.consume(tokens)


def rate_limit(key: str, tokens: int = 1, rate_limiter: Optional[RateLimiter] = None):
    """
    Decorator to apply rate limiting to a function.

    Args:
        key (str): The resource or operation key
        tokens (int): Number of tokens to consume
        rate_limiter (RateLimiter, optional): Rate limiter instance to use

    Returns:
        callable: Decorated function with rate limiting
    """
    # Create default rate limiter if none provided
    if rate_limiter is None:
        rate_limiter = RateLimiter()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply rate limiting
            rate_limiter.wait(key, tokens)

            # Call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


class ApiRateLimiter:
    """
    Specialized rate limiter for API endpoints with tiered limits.

    This class provides tiered rate limiting for API calls, allowing
    for different limits for different API endpoints and tiers.
    """

    def __init__(self):
        """Initialize the API rate limiter."""
        # Default tiers - can be customized based on API provider limits
        self.tiers = {
            "openai_chat": {
                "rate": 3500 / 60,  # 3500 RPM (gpt-4 limit)
                "capacity": 100,
            },
            "openai_embeddings": {
                "rate": 10000 / 60,  # 10000 RPM (embeddings limit)
                "capacity": 500,
            },
            "default": {"rate": settings.RATE_LIMIT / 60.0, "capacity": 10},
        }

        self.limiters: Dict[str, RateLimiter] = {}
        for tier, config in self.tiers.items():
            self.limiters[tier] = RateLimiter(
                default_rate=config["rate"], default_capacity=config["capacity"]
            )

    def limit_api_call(self, api_name: str, endpoint: str, tokens: int = 1):
        """
        Apply rate limiting to an API call.

        Args:
            api_name (str): Name of the API provider (e.g., "openai")
            endpoint (str): Specific endpoint or operation
            tokens (int): Number of tokens to consume
        """
        # Construct the tier key
        tier_key = f"{api_name}_{endpoint}"

        # Use the appropriate limiter or fall back to default
        if tier_key in self.limiters:
            limiter = self.limiters[tier_key]
        elif api_name in self.limiters:
            limiter = self.limiters[api_name]
        else:
            limiter = self.limiters["default"]

        # Apply the limit
        limiter.wait(endpoint, tokens)

    def limit_openai(self, endpoint: str, tokens: int = 1):
        """
        Apply rate limiting to an OpenAI API call.

        Args:
            endpoint (str): OpenAI endpoint ("chat", "embeddings", etc.)
            tokens (int): Number of tokens to consume
        """
        self.limit_api_call("openai", endpoint, tokens)


# Create singleton instances
general_limiter = RateLimiter()
api_limiter = ApiRateLimiter()


# Helper functions
def limit_api(api_name: str, endpoint: str, tokens: int = 1):
    """Apply rate limiting to an API call."""
    api_limiter.limit_api_call(api_name, endpoint, tokens)


def limit_openai(endpoint: str, tokens: int = 1):
    """Apply rate limiting to an OpenAI API call."""
    api_limiter.limit_openai(endpoint, tokens)
