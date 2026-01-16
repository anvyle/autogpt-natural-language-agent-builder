"""
Langfuse Integration Module

This module provides Langfuse integration for:
1. Tracing all LLM usage
2. Loading prompts dynamically from Langfuse
3. Running evals over time

Usage:
    from langfuse_integration import trace_llm_call, get_prompt
    
    # Trace LLM calls
    with trace_llm_call("decompose_description", inputs={"description": desc}):
        result = await llm.ainvoke(messages)
    
    # Load prompts from Langfuse
    prompt = get_prompt("DECOMPOSITION_PROMPT_TEMPLATE")
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
import asyncio

import config
from logging_config import get_logger

# Create module-specific logger
logger = get_logger(__name__)

# Try to import Langfuse
try:
    from langfuse import Langfuse, observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not available. Install with: pip install langfuse")

# =============================================================================
# GLOBAL LANGFUSE CLIENT
# =============================================================================

_langfuse_client: Optional[Any] = None
_langfuse_enabled: bool = False

def initialize_langfuse():
    """Initialize the Langfuse client with credentials from config."""
    global _langfuse_client, _langfuse_enabled
    
    if not LANGFUSE_AVAILABLE:
        logger.warning("Langfuse library not available. Tracing disabled.")
        _langfuse_enabled = False
        return
    
    if not config.is_langfuse_enabled():
        logger.info("Langfuse credentials not configured. Tracing disabled.")
        _langfuse_enabled = False
        return
    
    try:
        secret_key = config.get_langfuse_secret_key()
        public_key = config.get_langfuse_public_key()
        base_url = config.get_langfuse_base_url()
        
        _langfuse_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url
        )
        _langfuse_enabled = True
        logger.info(f"✅ Langfuse initialized successfully (base_url: {base_url})")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Langfuse: {e}")
        _langfuse_enabled = False

def is_langfuse_enabled() -> bool:
    """Check if Langfuse is enabled and initialized."""
    return _langfuse_enabled

def get_langfuse_client() -> Optional[Any]:
    """Get the Langfuse client instance."""
    return _langfuse_client

# =============================================================================
# TRACING DECORATORS
# =============================================================================

def trace_llm_call(name: str, **kwargs):
    """
    Context manager for tracing LLM calls with Langfuse.
    
    Args:
        name: Name of the LLM operation (e.g., "decompose_description")
        **kwargs: Additional metadata to log (e.g., inputs, model, temperature)
    
    Usage:
        with trace_llm_call("decompose", inputs={"description": desc}):
            result = await llm.ainvoke(messages)
    """
    if not _langfuse_enabled or not LANGFUSE_AVAILABLE:
        # If Langfuse is not enabled, return a no-op context manager
        class NoOpContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOpContext()
    
    try:
        # Use Langfuse's trace context
        return observe(name=name, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to create Langfuse trace: {e}")
        # Return no-op context on error
        class NoOpContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOpContext()

def trace_llm_function(name: Optional[str] = None):
    """
    Decorator for tracing entire functions with Langfuse.
    
    Args:
        name: Optional name for the trace. If not provided, uses function name.
    
    Usage:
        @trace_llm_function("decompose_description")
        async def decompose_description(desc):
            ...
    """
    def decorator(func: Callable):
        if not _langfuse_enabled or not LANGFUSE_AVAILABLE:
            # If Langfuse is not enabled, return the original function
            return func
        
        trace_name = name or func.__name__
        
        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await observe(name=trace_name)(func)(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Langfuse tracing failed for {trace_name}: {e}")
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return observe(name=trace_name)(func)(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Langfuse tracing failed for {trace_name}: {e}")
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator

# =============================================================================
# PROMPT MANAGEMENT
# =============================================================================

_prompt_cache: Dict[str, str] = {}

def get_prompt(
    prompt_name: str,
    fallback_prompt: Optional[str] = None,
    version: Optional[int] = None,
    variables: Optional[Dict[str, Any]] = None
) -> str:
    """
    Load a prompt from Langfuse by name with optional versioning.
    Falls back to provided fallback_prompt if Langfuse is unavailable or prompt not found.
    
    Args:
        prompt_name: Name of the prompt in Langfuse (e.g., "DECOMPOSITION_PROMPT_TEMPLATE")
        fallback_prompt: Fallback prompt to use if Langfuse is unavailable
        version: Optional specific version to load
        variables: Optional variables to compile the prompt with
    
    Returns:
        The prompt text, either from Langfuse or fallback
    """
    # If Langfuse is not enabled, use fallback
    if not _langfuse_enabled or not _langfuse_client:
        if fallback_prompt is None:
            logger.warning(f"Langfuse disabled and no fallback provided for prompt: {prompt_name}")
            return ""
        return fallback_prompt
    
    try:
        # Check cache first
        cache_key = f"{prompt_name}::{version or 'latest'}"
        if cache_key in _prompt_cache:
            logger.debug(f"Using cached prompt: {prompt_name}")
            prompt_text = _prompt_cache[cache_key]
        else:
            # Fetch from Langfuse
            logger.info(f"Fetching prompt from Langfuse: {prompt_name} (version: {version or 'latest'})")
            
            if version:
                prompt = _langfuse_client.get_prompt(prompt_name, version=version)
            else:
                prompt = _langfuse_client.get_prompt(prompt_name)
            
            if prompt:
                prompt_text = prompt.prompt
                if variables:
                    prompt_text = prompt.compile(**variables)
                
                # Cache the prompt
                _prompt_cache[cache_key] = prompt_text
                logger.info(f"✅ Successfully loaded prompt from Langfuse: {prompt_name}")
            else:
                logger.warning(f"Prompt not found in Langfuse: {prompt_name}, using fallback")
                prompt_text = fallback_prompt or ""
        
        return prompt_text
        
    except Exception as e:
        logger.error(f"❌ Error loading prompt from Langfuse ({prompt_name}): {e}")
        if fallback_prompt is None:
            logger.error(f"No fallback prompt available for: {prompt_name}")
            return ""
        logger.info(f"Using fallback prompt for: {prompt_name}")
        return fallback_prompt

def clear_prompt_cache():
    """Clear the prompt cache to force reload from Langfuse."""
    global _prompt_cache
    _prompt_cache = {}
    logger.info("Prompt cache cleared")

def refresh_prompt(prompt_name: str, version: Optional[int] = None) -> bool:
    """
    Force refresh a specific prompt from Langfuse.
    
    Args:
        prompt_name: Name of the prompt to refresh
        version: Optional specific version to load
    
    Returns:
        True if successful, False otherwise
    """
    cache_key = f"{prompt_name}::{version or 'latest'}"
    if cache_key in _prompt_cache:
        del _prompt_cache[cache_key]
    
    # Try to reload
    try:
        if version:
            prompt = _langfuse_client.get_prompt(prompt_name, version=version)
        else:
            prompt = _langfuse_client.get_prompt(prompt_name)
        
        if prompt:
            if hasattr(prompt, 'prompt'):
                prompt_text = prompt.prompt
            else:
                prompt_text = str(prompt)
            
            _prompt_cache[cache_key] = prompt_text
            logger.info(f"✅ Refreshed prompt: {prompt_name}")
            return True
        else:
            logger.warning(f"Prompt not found: {prompt_name}")
            return False
    except Exception as e:
        logger.error(f"❌ Error refreshing prompt {prompt_name}: {e}")
        return False

# =============================================================================
# SCORING AND EVALUATION
# =============================================================================

def score_generation(
    trace_id: str,
    name: str,
    value: float,
    comment: Optional[str] = None
):
    """
    Score a traced generation for evaluation purposes.
    
    Args:
        trace_id: The trace ID to score
        name: Name of the score (e.g., "quality", "accuracy")
        value: Score value (typically 0-1 or 0-100)
        comment: Optional comment about the score
    """
    if not _langfuse_enabled or not _langfuse_client:
        logger.debug("Langfuse not enabled, skipping scoring")
        return
    
    try:
        _langfuse_client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment
        )
        logger.debug(f"Scored trace {trace_id}: {name}={value}")
    except Exception as e:
        logger.error(f"Error scoring trace: {e}")

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize Langfuse on module import
initialize_langfuse()

