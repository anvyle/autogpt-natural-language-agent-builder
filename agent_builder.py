import json
import re
import logging
from datetime import datetime
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from utils import load_json_async, AgentFixer, AgentValidator

import config
from blocks_fetcher import fetch_and_cache_blocks, get_cache_info
from langfuse_integration import trace_llm_function, get_prompt, is_langfuse_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
OUTPUT_DIR = Path(f"generated_agents/{datetime.now().strftime('%Y%m%d')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-3-pro-preview"

# Legacy block file kept for reference
# BLOCK_FILE = "./data/blocks_2025_11_11_edited.json"
EXAMPLE_FILE = "./data/Resume_Rater_AI.json"

# =============================================================================
# GLOBAL STATE FOR BLOCKS
# =============================================================================

# Global variables to store blocks and block summaries
_blocks = None
_block_summaries = None
_blocks_loaded = False


# =============================================================================
# JSON PARSING UTILITIES
# =============================================================================

def _parse_llm_json_or_none(raw_text: str):
    """Try multiple strategies to extract and parse JSON from an LLM response.

    The model may wrap JSON in triple backticks, include language hints, or add
    surrounding prose. This helper attempts common variants and returns a parsed
    object or None.
    """
    if raw_text is None:
        return None

    text = str(raw_text)

    candidates = []

    # 1) Fenced code block ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    # 2) Raw text as-is
    candidates.append(text.strip())

    # 3) Heuristic: first {...} span
    lcurly = text.find("{")
    rcurly = text.rfind("}")
    if lcurly != -1 and rcurly != -1 and rcurly > lcurly:
        candidates.append(text[lcurly:rcurly + 1].strip())

    # 4) Heuristic: first [...] span
    lbrack = text.find("[")
    rbrack = text.rfind("]")
    if lbrack != -1 and rbrack != -1 and rbrack > lbrack:
        candidates.append(text[lbrack:rbrack + 1].strip())

    # Try to parse candidates in order
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    return None

# =============================================================================
# PROMPT GETTER FUNCTIONS
# =============================================================================

def get_decomposition_prompt(block_summaries: list) -> str:
    """Get the decomposition prompt with block summaries from Langfuse."""
    # Load from Langfuse (no fallback - prompts must be in Langfuse)
    template = get_prompt("DECOMPOSITION_PROMPT_TEMPLATE", variables={"block_summaries": json.dumps(block_summaries, indent=2)})
    return template

def get_agent_generation_prompt(used_blocks: list, example: str) -> str:
    """Get the agent generation prompt with used blocks and example from Langfuse."""
    # Load from Langfuse (no fallback - prompts must be in Langfuse)
    template = get_prompt("AGENT_GENERATION_PROMPT_TEMPLATE", variables={"used_blocks": json.dumps(used_blocks, indent=2), "example": example})
    return template

def get_incremental_update_system_prompt(block_summaries: list) -> str:
    """Get the incremental update system prompt with block summaries from Langfuse."""
    # Load from Langfuse (no fallback - prompts must be in Langfuse)
    template = get_prompt("INCREMENTAL_UPDATE_SYSTEM_PROMPT_TEMPLATE", variables={"block_summaries": json.dumps(block_summaries, indent=2)})
    return template

def get_incremental_update_human_prompt(improvement_request: str, current_instructions) -> str:
    """Get the incremental update human prompt with improvement request and current instructions."""
    # Convert current_instructions to string format for the prompt if it's JSON
    if isinstance(current_instructions, dict):
        instructions_text = json.dumps(current_instructions, indent=2)
    else:
        instructions_text = str(current_instructions)
    template = get_prompt("INCREMENTAL_UPDATE_HUMAN_PROMPT_TEMPLATE", variables={"improvement_request": improvement_request, "current_instructions": instructions_text})
    
    return template

def get_incremental_agent_update_system_prompt(used_blocks: list, example: str) -> str:
    """Get the incremental agent update system prompt with used blocks and example from Langfuse."""
    # Load from Langfuse (no fallback - prompts must be in Langfuse)
    template = get_prompt("INCREMENTAL_AGENT_UPDATE_SYSTEM_PROMPT_TEMPLATE", variables={"used_blocks": json.dumps(used_blocks, indent=2), "example": example})
    return template

def get_incremental_agent_update_human_prompt(current_agent_json: dict, updated_instructions: str) -> str:
    """Get the incremental agent update human prompt with current agent JSON and updated instructions."""
    template = get_prompt("INCREMENTAL_AGENT_UPDATE_HUMAN_PROMPT_TEMPLATE", variables={"current_agent_json": json.dumps(current_agent_json, indent=2), "updated_instructions": updated_instructions})
    return template

def get_patch_generation_system_prompt(block_summaries: list) -> str:
    """Get the patch generation system prompt with block summaries from Langfuse."""
    # Load from Langfuse (no fallback - prompts must be in Langfuse)
    template = get_prompt("PATCH_GENERATION_SYSTEM_PROMPT_TEMPLATE", variables={"block_summaries": json.dumps(block_summaries, indent=2)})
    return template

def get_patch_generation_human_prompt(agent_summary: dict, current_agent: dict, update_request: str) -> str:
    """Get the patch generation human prompt with agent context and update request."""
    template = get_prompt("PATCH_GENERATION_HUMAN_PROMPT_TEMPLATE", variables={"agent_summary": json.dumps(agent_summary, indent=2), "current_agent": json.dumps(current_agent, indent=2), "update_request": update_request})
    return template


async def initialize_blocks(force_refresh: bool = False):
    """
    Initialize blocks and block summaries by fetching from API with caching.
    This should be called once at application startup.
    
    Args:
        force_refresh: If True, bypass cache and fetch fresh blocks from API
    """
    global _blocks, _block_summaries, _blocks_loaded
    
    if _blocks_loaded and not force_refresh:
        logging.info("Blocks already loaded, skipping initialization")
        return
    
    try:
        # Log cache status
        cache_info = await get_cache_info()
        logging.info(f"Cache status: {cache_info.get('status', 'unknown')}")
        
        # Fetch blocks (from cache or API)
        logging.info("Loading blocks...")
        _blocks = await fetch_and_cache_blocks(force_refresh=force_refresh)
        
        _block_summaries = [
            {
                "id": block["id"],
                "name": block["name"],
                "description": block.get("description", ""),
                "inputs_schema": block.get("inputSchema", {}),
                "outputs_schema": block.get("outputSchema", {}),
            } for block in _blocks
        ]
        _blocks_loaded = True
        logging.info(f"✅ Successfully loaded {len(_blocks)} blocks")
        
        # Log updated cache info
        cache_info = await get_cache_info()
        if cache_info.get("status") == "fresh":
            logging.info(f"Using blocks from cache (age: {cache_info.get('age_hours', 0):.1f}h)")
        
    except Exception as e:
        logging.error(f"❌ Failed to load blocks: {e}")
        raise

def get_blocks():
    """Get the loaded blocks. Returns None if blocks haven't been initialized."""
    return _blocks

def get_block_summaries():
    """Get the loaded block summaries. Returns None if blocks haven't been initialized."""
    return _block_summaries

def is_blocks_loaded():
    """Check if blocks have been loaded."""
    return _blocks_loaded


@trace_llm_function("decompose_description")
async def decompose_description(description, original_text=None, user_instruction=None, retry_feedback=None):
    """
    Decompose a description into step-by-step instructions.
    
    Args:
        description: The goal or description to decompose
        original_text: Original instructions (for revision scenarios)
        user_instruction: User feedback for revision
        retry_feedback: Validation error feedback for retry scenarios
    
    Returns:
        Parsed JSON dict with instructions or None on error
    """
    logging.info(f"Decomposing description: \n{description}\n...")
    
    # Log to Langfuse if enabled
    if is_langfuse_enabled():
        logging.info("🔍 Langfuse tracing enabled for decompose_description")
    
    # Ensure blocks are loaded
    if not _blocks_loaded:
        logging.error("❌ Blocks not loaded. Call initialize_blocks() first.")
        return None
    
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

    if original_text and user_instruction:
        logging.info(f"Revising instructions based on user feedback...")
        
        # Convert original_text to string format for the prompt if it's JSON
        if isinstance(original_text, dict):
            original_text_str = json.dumps(original_text, indent=2)
        else:
            original_text_str = str(original_text)
        
        prompt = f"""
            You previously generated the following step-by-step instructions:

            ---
            {original_text_str}
            ---

            Now revise them based on this user feedback:
            "{user_instruction}"

            ---

            Update the steps accordingly.
            Output ONLY the updated instructions in JSON format, maintaining the same keys and structure.
            Do NOT include any explanatory text, only the JSON instructions.
            """
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            if response is None:
                logging.error("❌ No response received from LLM")
                return None
            
            parsed = _parse_llm_json_or_none(str(response.text))
            if parsed is None:
                logging.error("❌ Error revising instructions: Failed to parse JSON from LLM response")
                return None
            return parsed
        except Exception as e:
            logging.error(f"❌ Error revising instructions: {e}")
            return None

    if original_text and retry_feedback:
        logging.info(f"Revising instructions based on validation error: {retry_feedback}")
        
        # Convert original_text to string format for the prompt if it's JSON
        if isinstance(original_text, dict):
            original_text_str = json.dumps(original_text, indent=2)
        else:
            original_text_str = str(original_text)
        
        prompt = f"""
            Please update the instructions below to fix the noted validation error. 

            ---
            Previous step-by-step instructions:
            {original_text_str}
            ---

            Validation failed with this message: {retry_feedback}

            Revise the instructions to address the validation issue. 
            - Preserve the overall structure of the original instructions.
            - Add, remove, or modify steps only as needed to resolve the issue.
            - Output ONLY the updated instructions in JSON format, maintaining the same keys and structure.
            - Do NOT include any explanatory text, only the JSON instructions.

            You can refer to the following available blocks for implementation:
            {json.dumps(_block_summaries, indent=2)}
        """
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            if response is None:
                logging.error("❌ No response received from LLM")
                return None
            
            parsed = _parse_llm_json_or_none(str(response.text))
            if parsed is None:
                logging.error("❌ Error revising instructions: Failed to parse JSON from LLM response")
                return None
            return parsed
        except Exception as e:
            logging.error(f"❌ Error revising instructions: {e}")
            return None

    prompt = get_decomposition_prompt(_block_summaries)
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=description)
        ])
        if response is None:
            logging.error("❌ No response received from LLM")
            return None
        
        parsed = _parse_llm_json_or_none(str(response.text))
        if parsed is None:
            logging.error("❌ Error decomposing description: Failed to parse JSON from LLM response")
            return None
        return parsed
        
    except Exception as e:
        logging.error(f"❌ Error decomposing description: {e}")
        return None


@trace_llm_function("generate_agent_json")
async def generate_agent_json_from_subtasks(instructions):
    """
    Generate agent JSON from instructions with single retry for parsing failures
    and single retry for validation failures using patch-based updates.
    
    Args:
        instructions: Step-by-step instructions (dict or string)
    
    Returns:
        Tuple of (agent_json, error_message)
    """
    logging.info(f"Generating agent JSON from instructions...")
    
    # Log to Langfuse if enabled
    if is_langfuse_enabled():
        logging.info("🔍 Langfuse tracing enabled for generate_agent_json")
    
    # Ensure blocks are loaded
    if not _blocks_loaded:
        logging.error("❌ Blocks not loaded. Call initialize_blocks() first.")
        return None, "Blocks not loaded"

    # Extract block names from the structured JSON format
    block_names = set()
    
    steps = instructions.get("steps", [])
    for step in steps:
        block_name = step.get("block_name")
        if block_name:
            logging.info(f"Found block name: {block_name}")
            block_names.add(block_name)

    used_blocks = []
    for block in _blocks:
        block_name = block.get("name") or block.get("block_name")
        if block_name and block_name in block_names:
            used_blocks.append(block)

    if not used_blocks:
        used_blocks = _blocks

    example = await load_json_async(EXAMPLE_FILE)
    example = json.dumps(example, indent=2)
    
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
        prompt = get_agent_generation_prompt(used_blocks, example)

        # Ensure 'instructions' is a string before sending as LLM message content
        if isinstance(instructions, dict):
            instructions_content = json.dumps(instructions, indent=2)
        else:
            instructions_content = str(instructions)

        # Retry once for JSON parsing failures (2 total attempts)
        agent_json = None
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=instructions_content)
        ]
        
        response = await llm.ainvoke(messages)
        if response is None:
            logging.error("❌ No response received from LLM")
            return None, "No response received from LLM"
                
        agent_json = _parse_llm_json_or_none(str(response.text))  
        if agent_json is None:
            logging.error("❌ Error generating agent JSON: Failed to parse JSON from LLM response")
            return None, "Failed to parse JSON from LLM response"
        
        # Apply automatic fixes
        agent_fixer = AgentFixer()
        agent_json = await agent_fixer.apply_all_fixes(agent_json, _blocks)

        # Validate and use patch-based retry for validation failures (single retry)
        validator = AgentValidator()
        is_valid, error = validator.validate(agent_json, _blocks)
        
        if not is_valid:
            logging.warning(f"⚠️ Initial validation failed: {error}")
            logging.info("🔧 Attempting patch-based fix for validation errors...")
            
            # Generate patch to fix validation errors
            patch_request = f"""Fix the following validation errors in the agent:

            **Validation Error:**
            {error}

            **Instructions:**
            Please generate a minimal patch to fix only these validation errors while preserving all other parts of the agent exactly as they are."""
            
            patch_result, patch_error = await generate_agent_patch(patch_request, agent_json)
            
            if not patch_error and patch_result:
                # Check if it's clarifying questions (shouldn't happen, but handle it)
                if not (isinstance(patch_result, dict) and patch_result.get("type") == "clarifying_questions"):
                    # Apply the patch
                    fixed_agent, apply_error = apply_agent_patch(agent_json, patch_result)
                    
                    if not apply_error and fixed_agent:
                        # Apply automatic fixes again after patching
                        fixed_agent = await agent_fixer.apply_all_fixes(fixed_agent, _blocks)
                        
                        # Validate the fixed agent
                        is_valid, error = validator.validate(fixed_agent, _blocks)
                        
                        if is_valid:
                            logging.info("✅ Validation errors fixed with patch-based approach!")
                            agent_json = fixed_agent
                        else:
                            logging.warning(f"⚠️ Validation still failing after patch: {error}")
                    else:
                        logging.warning(f"⚠️ Failed to apply fix patch: {apply_error}")
                else:
                    logging.warning("⚠️ Patch generation returned clarifying questions - cannot auto-fix")
            else:
                logging.warning(f"⚠️ Failed to generate fix patch: {patch_error}")
            
            # Final validation check
            if not is_valid:
                logging.error(f"❌ Validation failed after patch-based fix attempt: {error}")
                return None, error

        # Success - agent generated and validated
        filename = agent_json["name"].replace(" ", "_")
        agent_json_path = OUTPUT_DIR / f"{filename}.json"
        try:
            with open(agent_json_path, "w", encoding="utf-8") as f:
                json.dump(agent_json, f, indent=2, ensure_ascii=False)
            logging.info(f"✅ Saved agent.json to: {agent_json_path}")
        except Exception as e:
            logging.error(f"❌ Failed to save agent.json: {e}")
            
        return agent_json, None
        
    except Exception as e:
        logging.error(f"❌ Error during agent generation: {e}")
        return None, f"Error during agent generation: {e}"


@trace_llm_function("generate_agent_patch")
async def generate_agent_patch(update_request: str, current_agent: dict):
    """
    Generate a minimal JSON patch to update the agent.
    Can also return clarifying questions if more information is needed.
    
    Args:
        update_request: User's natural language update request
        current_agent: Current agent JSON
    
    Returns:
        Tuple of (patch_dict_or_questions, error_message)
        patch_dict_or_questions can be:
        - {"type": "clarifying_questions", "questions": [...]} if more info needed
        - {"intent": {...}, "patches": [...]} for actual patches
    """
    logging.info(f"Generating agent patch for request: {update_request}")
    
    # Log to Langfuse if enabled
    if is_langfuse_enabled():
        logging.info("🔍 Langfuse tracing enabled for generate_agent_patch")
    
    # Ensure blocks are loaded
    if not _blocks_loaded:
        logging.error("❌ Blocks not loaded. Call initialize_blocks() first.")
        return None, "Blocks not loaded"
    
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    
    # Create a readable representation of current agent
    agent_summary = {
        "name": current_agent.get("name"),
        "description": current_agent.get("description"),
        "nodes": []
    }
    
    for node in current_agent.get('nodes', []):
        agent_summary["nodes"].append({
            "id": node.get('id'),
            "block_id": node.get('block_id'),
            "customized_name": node.get('metadata', {}).get('customized_name', 'Unnamed'),
            "position": node.get('metadata', {}).get('position'),
            "input_default": node.get('input_default', {})
        })
    
    # Use getter functions for prompts
    system_prompt = get_patch_generation_system_prompt(_block_summaries)
    human_prompt = get_patch_generation_human_prompt(agent_summary, current_agent, update_request)
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        if response is None:
            logging.error("❌ No response received from LLM")
            return None, "No response received from LLM"
        
        result = _parse_llm_json_or_none(str(response.text))
        if result is None:
            logging.error("❌ Error generating patch: Failed to parse JSON from LLM response")
            return None, "Failed to parse JSON from LLM response"
        
        # Check if it's clarifying questions
        if isinstance(result, dict) and result.get("type") == "clarifying_questions":
            logging.info("📋 LLM returned clarifying questions for patch generation")
            return result, None
        
        return result, None
        
    except Exception as e:
        logging.error(f"❌ Error generating patch: {e}")
        return None, f"Error generating patch: {e}"


def apply_agent_patch(current_agent: dict, patch: dict) -> tuple:
    """
    Apply a patch to the current agent, preserving all unchanged parts.
    
    Args:
        current_agent: Current agent JSON
        patch: Patch dict with intent and patches list
    
    Returns:
        Tuple of (updated_agent, error_message)
    """
    logging.info("Applying patch to agent...")
    
    try:
        # Deep copy to avoid mutating original
        import copy
        updated_agent = copy.deepcopy(current_agent)
        
        patches = patch.get('patches', [])
        
        for patch_item in patches:
            patch_type = patch_item.get('type')
            
            if patch_type == 'modify':
                # Modify existing node
                node_id = patch_item.get('node_id')
                changes = patch_item.get('changes', {})
                
                for node in updated_agent['nodes']:
                    if node['id'] == node_id:
                        # Apply changes recursively
                        _deep_update(node, changes)
                        logging.info(f"✓ Modified node {node_id}")
                        break
            
            elif patch_type == 'add':
                # Add new nodes and links
                new_nodes = patch_item.get('new_nodes', [])
                new_links = patch_item.get('new_links', [])
                
                # Inherit graph_id and graph_version
                graph_id = updated_agent.get('id')
                graph_version = updated_agent.get('version', 1)
                
                for new_node in new_nodes:
                    if new_node.get('graph_id') == 'inherit':
                        new_node['graph_id'] = graph_id
                    if new_node.get('graph_version') == 'inherit':
                        new_node['graph_version'] = graph_version
                    
                    updated_agent['nodes'].append(new_node)
                    logging.info(f"✓ Added node {new_node.get('id')}")
                
                updated_agent['links'].extend(new_links)
                logging.info(f"✓ Added {len(new_links)} link(s)")
            
            elif patch_type == 'delete':
                # Remove nodes and their links
                node_ids_to_delete = patch_item.get('node_ids', [])
                
                # Remove nodes
                updated_agent['nodes'] = [
                    node for node in updated_agent['nodes']
                    if node['id'] not in node_ids_to_delete
                ]
                
                # Remove associated links
                updated_agent['links'] = [
                    link for link in updated_agent['links']
                    if link.get('source_id') not in node_ids_to_delete
                    and link.get('sink_id') not in node_ids_to_delete
                ]
                
                logging.info(f"✓ Deleted {len(node_ids_to_delete)} node(s)")
                
                # Handle reconnection if specified
                reconnect = patch_item.get('reconnect')
                if reconnect and reconnect.get('maintain_data_flow'):
                    # Create new link to maintain flow
                    # This would need more sophisticated logic
                    pass
            
            elif patch_type == 'replace':
                # Replace node
                node_id = patch_item.get('node_id')
                new_node = patch_item.get('new_node')
                update_links = patch_item.get('update_links', [])
                
                # Replace the node
                for i, node in enumerate(updated_agent['nodes']):
                    if node['id'] == node_id:
                        # Preserve position if not specified
                        if 'position' not in new_node.get('metadata', {}):
                            new_node.setdefault('metadata', {})['position'] = node.get('metadata', {}).get('position')
                        
                        updated_agent['nodes'][i] = new_node
                        logging.info(f"✓ Replaced node {node_id}")
                        break
                
                # Update affected links
                for link_update in update_links:
                    link_id = link_update.get('link_id')
                    link_changes = link_update.get('changes', {})
                    
                    for link in updated_agent['links']:
                        if link['id'] == link_id:
                            _deep_update(link, link_changes)
                            break
        
        return updated_agent, None
        
    except Exception as e:
        logging.error(f"❌ Error applying patch: {e}")
        return None, f"Error applying patch: {e}"


def _deep_update(target: dict, updates: dict):
    """
    Deep update a dictionary, merging nested dicts recursively.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


async def update_agent_json_incrementally(update_request: str, current_agent_json: dict):
    """
    Update agent JSON using patch-based incremental updates.
    This preserves unchanged parts exactly and only modifies what's necessary.
    Supports clarifying questions, validation, fixing, and single retry on failure.
    
    Args:
        update_request: User's natural language update request (e.g., "Add error handling to step 3")
        current_agent_json: Current agent JSON to update
    
    Returns:
        Tuple of (result, error_message)
        result can be:
        - {"type": "clarifying_questions", "questions": [...]} if more info needed
        - updated_agent_json dict if successful
        - None if failed
    """
    logging.info(f"Updating agent incrementally with request: {update_request}")
    
    # Log to Langfuse if enabled
    if is_langfuse_enabled():
        logging.info("🔍 Langfuse tracing enabled for update_agent_incrementally")
    
    # Ensure blocks are loaded
    if not _blocks_loaded:
        logging.error("❌ Blocks not loaded. Call initialize_blocks() first.")
        return None, "Blocks not loaded"
    
    try:
        
        # Retry once for patch generation and application (2 total attempts)
        for attempt in range(2):
            if attempt > 0:
                logging.info(f"🔄 Retry attempt for agent update")
            
            # Step 1: Generate the patch (may return clarifying questions)
            result, error = await generate_agent_patch(
                update_request,
                current_agent_json
            )
            
            if error:
                if attempt < 1:
                    logging.warning(f"⚠️ Patch generation failed: {error}. Retrying...")
                    continue
                return None, error
            
            if not result:
                if attempt < 1:
                    logging.warning(f"⚠️ No patch generated. Retrying...")
                    continue
                return None, "Failed to generate patch"
            
            # Check if LLM returned clarifying questions
            if isinstance(result, dict) and result.get("type") == "clarifying_questions":
                logging.info("📋 Returning clarifying questions to user")
                return result, None
            
            # Step 2: Apply the patch
            logging.info(f"Applying patch with {len(result.get('patches', []))} operations")
            updated_agent, error = apply_agent_patch(current_agent_json, result)
            
            if error:
                if attempt < 1:
                    logging.warning(f"⚠️ Patch application failed: {error}. Retrying...")
                    update_request = f"{update_request}\n\nPrevious attempt failed with error: {error}\nPlease fix this issue."
                    continue
                return None, error
            
            if not updated_agent:
                if attempt < 1:
                    logging.warning(f"⚠️ Patch application returned no agent. Retrying...")
                    continue
                return None, "Failed to apply patch"
            
            # Step 3: Fix any issues
            agent_fixer = AgentFixer()
            updated_agent = await agent_fixer.apply_all_fixes(updated_agent, _blocks)
            
            fixes_applied = agent_fixer.get_fixes_applied()
            if fixes_applied:
                logging.info(f"🔧 Applied {len(fixes_applied)} automatic fixes to patched agent")
            
            # Step 4: Validate the result
            validator = AgentValidator()
            is_valid, validation_error = validator.validate(updated_agent, _blocks)
            
            if not is_valid:
                if attempt < 1:
                    logging.warning(f"⚠️ Validation failed: {validation_error}. Retrying with feedback...")
                    # Enhance update request with validation feedback for next retry
                    update_request = f"{update_request}\n\n**Validation Error from Previous Attempt:**\n{validation_error}\n\nPlease generate a patch that addresses these validation errors."
                    continue
                return None, validation_error
            
            # Success!
            logging.info("✅ Agent updated successfully with patch-based system")
            return updated_agent, None
        
        # Should not reach here, but just in case
        return None, "Failed to update agent after 2 attempts"
        
    except Exception as e:
        logging.error(f"❌ Error during patch-based agent update: {e}")
        return None, f"Error updating agent: {e}"


