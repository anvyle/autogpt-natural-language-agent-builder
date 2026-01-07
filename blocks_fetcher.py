"""
Blocks fetcher module for AutoGPT Agent Builder.
Handles dynamic fetching of blocks from the AutoGPT platform API with caching.
"""

import json
import logging
import aiofiles
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path("./data")
CACHE_FILE = CACHE_DIR / "blocks_cache.json"
CACHE_METADATA_FILE = CACHE_DIR / "blocks_cache_metadata.json"
CACHE_MAX_AGE_HOURS = 24  # Cache valid for 24 hours

# Fallback to hard-coded file if API fails
FALLBACK_BLOCK_FILE = "./data/blocks_2025_11_11_edited.json"


async def fetch_blocks_from_api(api_key: str, api_url: str, timeout: int = 120) -> List[Dict[str, Any]]:
    """
    Fetch blocks from the AutoGPT platform API.
    
    Args:
        api_key: API key for authentication
        api_url: URL of the blocks API endpoint
        timeout: Request timeout in seconds (default: 120s for large response)
    
    Returns:
        List of block dictionaries
        
    Raises:
        Exception: If the API request fails
    """
    logger.info(f"Fetching blocks from API: {api_url}")
    
    headers = {
        "X-API-Key": api_key,  # Try X-API-Key header format
        "Content-Type": "application/json"
    }
    
    # Use a longer timeout for the large response
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(api_url, headers=headers) as response:
                if response.status == 200:
                    blocks = await response.json()
                    logger.info(f"✅ Successfully fetched {len(blocks)} blocks from API")
                    return blocks
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"API request failed with status {response.status}: {error_text}"
                    )
    except aiohttp.ClientError as e:
        raise Exception(f"Network error while fetching blocks: {e}")
    except Exception as e:
        raise Exception(f"Error fetching blocks from API: {e}")


async def save_blocks_to_cache(blocks: List[Dict[str, Any]]) -> None:
    """
    Save blocks to local cache file.
    
    Args:
        blocks: List of block dictionaries to cache
    """
    try:
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save blocks
        async with aiofiles.open(CACHE_FILE, 'w') as f:
            await f.write(json.dumps(blocks, indent=2))
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "blocks_count": len(blocks),
            "source": "api"
        }
        async with aiofiles.open(CACHE_METADATA_FILE, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        logger.info(f"✅ Cached {len(blocks)} blocks to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to save blocks to cache: {e}")


async def load_blocks_from_cache() -> Optional[List[Dict[str, Any]]]:
    """
    Load blocks from local cache file if it exists and is recent enough.
    
    Returns:
        List of block dictionaries or None if cache is invalid/missing
    """
    try:
        if not CACHE_FILE.exists() or not CACHE_METADATA_FILE.exists():
            logger.info("No cache file found")
            return None
        
        # Check cache age
        async with aiofiles.open(CACHE_METADATA_FILE, 'r') as f:
            metadata_text = await f.read()
            metadata = json.loads(metadata_text)
        
        cache_timestamp = datetime.fromisoformat(metadata["timestamp"])
        cache_age = datetime.now() - cache_timestamp
        
        if cache_age > timedelta(hours=CACHE_MAX_AGE_HOURS):
            logger.info(f"Cache is {cache_age.total_seconds() / 3600:.1f} hours old (max: {CACHE_MAX_AGE_HOURS}h), will refresh")
            return None
        
        # Load cached blocks
        async with aiofiles.open(CACHE_FILE, 'r') as f:
            blocks_text = await f.read()
            blocks = json.loads(blocks_text)
        
        logger.info(f"✅ Loaded {len(blocks)} blocks from cache (age: {cache_age.total_seconds() / 3600:.1f}h)")
        return blocks
        
    except Exception as e:
        logger.warning(f"Failed to load blocks from cache: {e}")
        return None


async def load_blocks_from_fallback() -> List[Dict[str, Any]]:
    """
    Load blocks from the fallback hard-coded file.
    
    Returns:
        List of block dictionaries
        
    Raises:
        Exception: If fallback file cannot be loaded
    """
    logger.warning(f"Loading blocks from fallback file: {FALLBACK_BLOCK_FILE}")
    
    try:
        async with aiofiles.open(FALLBACK_BLOCK_FILE, 'r') as f:
            blocks_text = await f.read()
            blocks = json.loads(blocks_text)
        
        logger.info(f"✅ Loaded {len(blocks)} blocks from fallback file")
        return blocks
    except Exception as e:
        raise Exception(f"Failed to load fallback blocks file: {e}")


async def fetch_and_cache_blocks(
    force_refresh: bool = False,
    use_fallback_on_error: bool = True
) -> List[Dict[str, Any]]:
    """
    Main function to fetch blocks with caching strategy.
    
    Strategy:
    1. If not force_refresh, try to load from cache
    2. If cache miss or force_refresh, fetch from API
    3. Cache the API response
    4. On API error, use cached version if available
    5. On all failures, fall back to hard-coded file if use_fallback_on_error is True
    
    Args:
        force_refresh: If True, bypass cache and fetch from API
        use_fallback_on_error: If True, use fallback file on all errors
    
    Returns:
        List of block dictionaries
        
    Raises:
        Exception: If all methods fail
    """
    # Try cache first (unless force refresh)
    if not force_refresh:
        cached_blocks = await load_blocks_from_cache()
        if cached_blocks is not None:
            return cached_blocks
    
    # Try to fetch from API
    api_key = config.get_autogpt_api_key()
    api_url = config.get_autogpt_blocks_api_url()
    
    if not api_key:
        logger.warning("⚠️  AUTOGPT_API_KEY not configured, skipping API fetch")
    else:
        try:
            blocks = await fetch_blocks_from_api(api_key, api_url)
            
            # Cache the result
            await save_blocks_to_cache(blocks)
            
            return blocks
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch blocks from API: {e}")
            
            # Try to use stale cache as fallback
            if CACHE_FILE.exists():
                logger.info("Attempting to use stale cache as fallback...")
                try:
                    async with aiofiles.open(CACHE_FILE, 'r') as f:
                        blocks_text = await f.read()
                        blocks = json.loads(blocks_text)
                    logger.info(f"✅ Using stale cache with {len(blocks)} blocks")
                    return blocks
                except Exception as cache_error:
                    logger.error(f"Failed to load stale cache: {cache_error}")
    
    # Final fallback to hard-coded file
    if use_fallback_on_error:
        return await load_blocks_from_fallback()
    
    raise Exception("Failed to fetch blocks from all sources")


async def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache status.
    
    Returns:
        Dictionary with cache metadata
    """
    try:
        if not CACHE_METADATA_FILE.exists():
            return {
                "status": "no_cache",
                "message": "No cache file exists"
            }
        
        async with aiofiles.open(CACHE_METADATA_FILE, 'r') as f:
            metadata_text = await f.read()
            metadata = json.loads(metadata_text)
        
        cache_timestamp = datetime.fromisoformat(metadata["timestamp"])
        cache_age = datetime.now() - cache_timestamp
        is_fresh = cache_age <= timedelta(hours=CACHE_MAX_AGE_HOURS)
        
        return {
            "status": "fresh" if is_fresh else "stale",
            "timestamp": metadata["timestamp"],
            "age_hours": cache_age.total_seconds() / 3600,
            "blocks_count": metadata.get("blocks_count", 0),
            "source": metadata.get("source", "unknown"),
            "max_age_hours": CACHE_MAX_AGE_HOURS
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

