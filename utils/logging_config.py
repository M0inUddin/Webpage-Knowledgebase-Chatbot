"""
Logging configuration for the Manzil Chatbot application.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import json

from config import settings

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Generate log filename with timestamp
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"manzil_chatbot_{current_time}.log"
log_filepath = logs_dir / log_filename


# Create a JSON formatter for structured logging
class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add exception info if available
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Add extra fields if available
        if hasattr(record, "extra"):
            log_record.update(record.extra)

        return json.dumps(log_record)


def setup_logging():
    """Configure application logging based on settings."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_filepath)

    # Create formatters
    if settings.DEBUG_MODE:
        # Detailed formatter for debug mode
        console_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)

        # JSON formatter for file logs
        file_handler.setFormatter(JsonFormatter())
    else:
        # Simpler format for production
        console_format = "%(asctime)s - %(levelname)s - %(message)s"
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)

        # JSON formatter for file logs
        file_handler.setFormatter(JsonFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Create module loggers with appropriate levels
    loggers = {
        "crawler": log_level,
        "knowledge_base": log_level,
        "nlp": log_level,
        "rag": log_level,
        "ui": log_level,
        "main": log_level,
    }

    # Set up module loggers
    for logger_name, logger_level in loggers.items():
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(logger_level)

    # Log initial configuration
    logging.info(f"Logging initialized at level {settings.LOG_LEVEL}")
    logging.info(f"Log file: {log_filepath}")

    if settings.DEBUG_MODE:
        logging.info("Running in DEBUG mode")
        logging.debug(f"Configuration: {settings.get_config_summary()}")

    return root_logger


def get_logger(name):
    """Get a logger for a specific module."""
    return logging.getLogger(name)
