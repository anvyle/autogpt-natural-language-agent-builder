"""
Centralized logging configuration for the application.

This module provides a consistent logging setup that works with:
- Streamlit applications
- Langfuse tracing
- General application logging

Usage:
    from logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("This will show in the console")
"""

import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configure logging for the entire application.
    
    This function ensures that logging works properly even when
    running under Streamlit, which initializes logging before our code runs.
    
    Args:
        level: The logging level for the handler (default: logging.INFO)
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not root_logger.handlers:
        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter with timestamp, logger name, level, and message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger.addHandler(console_handler)
        # Set root logger to INFO to avoid DEBUG spam from third-party libraries
        root_logger.setLevel(logging.INFO)
    
    # Configure specific loggers for our application
    logging.getLogger("langfuse").setLevel(logging.INFO)
    logging.getLogger("agent_builder").setLevel(logging.INFO)
    logging.getLogger("utils").setLevel(logging.INFO)
    logging.getLogger("streamlit_agent_builder").setLevel(logging.INFO)
    
    # Suppress noisy third-party library loggers
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name):
    """
    Get a logger for the specified module.
    
    Args:
        name: The logger name (typically __name__)
        
    Returns:
        A configured logger instance
    """
    # Ensure logging is set up
    setup_logging()
    
    return logging.getLogger(name)

