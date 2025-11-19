import json
import logging
import aiofiles
import uuid
import re
from typing import Tuple, List, Dict, Any, Optional

async def load_json_async(file_path: str):
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    return json.loads(content)

class AgentFixer:
    """
    A comprehensive fixer for AutoGPT agents that applies various fixes to ensure
    agents are valid and functional.
    """
    
    def __init__(self):
        self.UUID_REGEX = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}$")
        self.DOUBLE_CURLY_BRACES_BLOCK_IDS = [
            "44f6c8ad-d75c-4ae1-8209-aad1c0326928",
            "6ab085e2-20b3-4055-bc3e-08036e01eca6",
            "90f8c45e-e983-4644-aa0b-b4ebe2f531bc",
            "363ae599-353e-4804-937e-b2ee3cef3da4",
            "3b191d9f-356f-482d-8238-ba04b6d18381",
            "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
            "3a7c4b8d-6e2f-4a5d-b9c1-f8d23c5a9b0e",
            "ed1ae7a0-b770-4089-b520-1f0005fad19a",
            "a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa",
            "b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0b1",
            "716a67b3-6760-42e7-86dc-18645c6e00fc",
            "530cf046-2ce0-4854-ae2c-659db17c7a46",
            "ed55ac19-356e-4243-a6cb-bc599e9b716f",
            "1f292d4a-41a4-4977-9684-7c8d560b9f91",
            "32a87eab-381e-4dd4-bdb8-4c47151be35a"
        ]
        self.FIX_VALUE2_EMPTY_STRING_BLOCK_IDS = ["715696a0-e1da-45c8-b209-c2fa9c3b0be6"]
        self.ADDTOLIST_BLOCK_ID = "aeb08fc1-2fc1-4141-bc8e-f758f183a822"
        self.ADDTODICTIONARY_BLOCK_ID = "31d1064e-7446-4693-a7d4-65e5ca1180d1"
        self.CODE_EXECUTION_BLOCK_ID = "0b02b072-abe7-11ef-8372-fb5d162dd712"
        self.DATA_SAMPLING_BLOCK_ID = "4a448883-71fa-49cf-91cf-70d793bd7d87"
        self.STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"
        self.UNIVERSAL_TYPE_CONVERTER_BLOCK_ID = "95d1b990-ce13-4d88-9737-ba5c2070c97b"
        self.fixes_applied = []
    
    def is_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        return isinstance(value, str) and self.UUID_REGEX.match(value) is not None
    
    def generate_uuid(self) -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())
    
    def add_fix_log(self, fix_description: str):
        """Add a fix description to the applied fixes list."""
        self.fixes_applied.append(fix_description)
    
    async def fix_agent_ids(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix agent and link IDs to ensure they are valid UUIDs.
        
        Args:
            agent: The agent dictionary to fix
            
        Returns:
            The fixed agent dictionary
        """
        
        def get_new_id() -> str:
            new_id = self.generate_uuid()
            return new_id

        # Fix link IDs
        links = []
        for link in agent.get("links", []):
            if not self.is_uuid(link["id"]):
                link["id"] = get_new_id()
                self.add_fix_log(f"Fixed link ID: {link['id']}")
            links.append(link)
        agent["links"] = links

        # Fix agent ID
        if not self.is_uuid(agent.get("id", "")):
            old_id = agent.get("id", "missing")
            agent["id"] = self.generate_uuid()
            self.add_fix_log(f"Fixed agent ID: {old_id} -> {agent['id']}")

        return agent
    
    async def fix_storevalue_before_condition(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a StoreValueBlock before each ConditionBlock to provide a value for 'value2'.

        - Creates a StoreValueBlock node with default input and data False
        - Adds a link from the StoreValueBlock 'output' to the ConditionBlock 'value2'
        - Skips if a link to 'value2' already exists for the ConditionBlock
        - Prevents duplicate StoreValueBlocks by checking if one already exists for the same condition

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """

        nodes = agent.get("nodes", [])
        links = agent.get("links", []) or []

        # Collect ConditionBlock node ids
        condition_block_id = self.FIX_VALUE2_EMPTY_STRING_BLOCK_IDS[0]
        condition_node_ids = {
            node.get("id") for node in nodes if node.get("block_id") == condition_block_id
        }

        if not condition_node_ids:
            return agent

        new_links = []
        nodes_to_add = []
        store_node_counter = 0  # Counter to ensure unique positions

        for link in links:
            # Identify links going into ConditionBlock.value2
            if link.get("sink_id") in condition_node_ids and link.get("sink_name") == "value2":
                condition_node_id = link.get("sink_id")

                # If the upstream source is already a StoreValueBlock.output, keep as-is
                source_node = next((n for n in nodes if n.get("id") == link.get("source_id")), None)
                if source_node and source_node.get("block_id") == self.STORE_VALUE_BLOCK_ID and link.get("source_name") == "output":
                    new_links.append(link)
                    continue

                # Check if there's already a StoreValueBlock connected to this condition's value2
                # This prevents duplicates when the fix runs multiple times
                existing_storevalue_for_condition = False
                for existing_link in links:
                    if (existing_link.get("sink_id") == condition_node_id and 
                        existing_link.get("sink_name") == "value2"):
                        existing_source_node = next((n for n in nodes if n.get("id") == existing_link.get("source_id")), None)
                        if (existing_source_node and 
                            existing_source_node.get("block_id") == self.STORE_VALUE_BLOCK_ID and 
                            existing_link.get("source_name") == "output"):
                            existing_storevalue_for_condition = True
                            break

                if existing_storevalue_for_condition:
                    self.add_fix_log(f"Skipped adding StoreValueBlock for ConditionBlock {condition_node_id} - already has one connected")
                    new_links.append(link)
                    continue

                # Create StoreValueBlock node (input will be linked; data left default None)
                store_node_id = self.generate_uuid()
                store_node = {
                    "id": store_node_id,
                    "block_id": self.STORE_VALUE_BLOCK_ID,
                    "input_default": {
                        "data": None
                    },
                    "metadata": {
                        "position": {
                            "x": store_node_counter * 200,  # Space nodes horizontally
                            "y": -100  # Position above the condition block
                        }
                    },
                    "graph_id": agent.get("id"),
                    "graph_version": 1
                }
                nodes_to_add.append(store_node)
                store_node_counter += 1

                # Rewire: old source -> StoreValueBlock.input
                upstream_to_store_link = {
                    "id": self.generate_uuid(),
                    "source_id": link.get("source_id"),
                    "source_name": link.get("source_name"),
                    "sink_id": store_node_id,
                    "sink_name": "input"
                }

                # Then StoreValueBlock.output -> ConditionBlock.value2
                store_to_condition_link = {
                    "id": self.generate_uuid(),
                    "source_id": store_node_id,
                    "source_name": "output",
                    "sink_id": condition_node_id,
                    "sink_name": "value2"
                }

                new_links.append(upstream_to_store_link)
                new_links.append(store_to_condition_link)

                self.add_fix_log(
                    f"Inserted StoreValueBlock {store_node_id} between {link.get('source_id')}:{link.get('source_name')} and ConditionBlock {condition_node_id} value2"
                )
            else:
                new_links.append(link)

        if nodes_to_add:
            nodes.extend(nodes_to_add)
            agent["nodes"] = nodes
            agent["links"] = new_links

        return agent

    async def fix_double_curly_braces(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix single curly braces to double curly braces in specific block types.
        Only fixes values that come from other blocks through links, not code content.
        Skips fixing if the block's output will be passed to a CodeExecutionBlock.
        
        Args:
            agent: The agent dictionary to fix
            
        Returns:
            The fixed agent dictionary
        """
        
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        
        for node in nodes:
            block_id = node.get("block_id")
            if block_id not in self.DOUBLE_CURLY_BRACES_BLOCK_IDS:
                continue

            node_id = node.get("id")
            input_data = node.get("input_default", {})

            # Check if this block's output is linked to a CodeExecutionBlock
            is_linked_to_code_execution = False
            for link in links:
                if link.get("source_id") == node_id:
                    # Find the sink node
                    sink_node = next((n for n in nodes if n.get("id") == link.get("sink_id")), None)
                    if sink_node and sink_node.get("block_id") == self.CODE_EXECUTION_BLOCK_ID:
                        is_linked_to_code_execution = True
                        break
            
            # Skip fixing if this block's output goes to a CodeExecutionBlock
            if is_linked_to_code_execution:
                continue

            for key in ("prompt", "format"):
                if key in input_data:
                    original_text = input_data[key]

                    # Avoid fixing already double-braced values
                    # Only fix simple variable names, not formatting logic
                    fixed_text = re.sub(r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})', r'{{\1}}', original_text)

                    if fixed_text != original_text:
                        input_data[key] = fixed_text
                        self.add_fix_log(f"Fixed {key} in block_id {block_id}: {original_text} -> {fixed_text}")

        return agent
    
    async def fix_addtolist_blocks(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix AddToList blocks by adding a prerequisite empty AddToList block.
        
        When an AddToList block is found, this fixer:
        1. Checks if there's a CreateListBlock before it (directly or through StoreValueBlock)
        2. If CreateListBlock exists (direct link), removes it and its link to AddToList block
        3. If CreateListBlock + StoreValueBlock exists, only removes the link from StoreValueBlock to AddToList block
        4. Adds an empty AddToList block before the original AddToList block
        5. The first block is standalone (not connected to other blocks)
        6. The second block receives input from previous blocks and can self-reference
        7. Ensures the original AddToList block has a self-referencing link
        8. Prevents duplicate prerequisite blocks by checking existing connections
        
        Args:
            agent: The agent dictionary to fix
            
        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        new_nodes = []
        new_links = []
        original_addtolist_node_ids = set()  # Track original AddToList node IDs
        createlist_block_id = "a912d5c7-6e00-4542-b2a9-8034136930e4"
        
        # First pass: identify CreateListBlock nodes and links that need to be removed
        createlist_nodes_to_remove = set()
        links_to_remove = []
        
        for link in links:
            source_node = next((node for node in nodes if node.get("id") == link.get("source_id")), None)
            sink_node = next((node for node in nodes if node.get("id") == link.get("sink_id")), None)
            
            # Case 1: CreateListBlock directly linked to AddToList block - remove both node and link
            if (source_node and sink_node and 
                source_node.get("block_id") == createlist_block_id and 
                sink_node.get("block_id") == self.ADDTOLIST_BLOCK_ID):
                createlist_nodes_to_remove.add(source_node.get("id"))
                links_to_remove.append(link)
                self.add_fix_log(f"Identified CreateListBlock {source_node.get('id')} linked to AddToList block {sink_node.get('id')} for removal")
            
            # Case 2: StoreValueBlock linked to AddToList block - check if there's a CreateListBlock before it
            if (source_node and sink_node and 
                source_node.get("block_id") == self.STORE_VALUE_BLOCK_ID and 
                sink_node.get("block_id") == self.ADDTOLIST_BLOCK_ID):
                # Check if this StoreValueBlock receives input from a CreateListBlock
                storevalue_id = source_node.get("id")
                has_createlist_before = False
                for prev_link in links:
                    if prev_link.get("sink_id") == storevalue_id:
                        prev_source_node = next((node for node in nodes if node.get("id") == prev_link.get("source_id")), None)
                        if prev_source_node and prev_source_node.get("block_id") == createlist_block_id:
                            has_createlist_before = True
                            break
                
                # If there's a CreateListBlock before StoreValueBlock, only remove the StoreValueBlock -> AddToList link
                if has_createlist_before:
                    links_to_remove.append(link)
                    self.add_fix_log(f"Identified StoreValueBlock {storevalue_id} (with CreateListBlock before it) linked to AddToList block {sink_node.get('id')} - removing only the link")
        
        # Second pass: process nodes, skipping CreateListBlock nodes that will be removed
        prerequisite_counter = 0  # Counter to ensure unique positions
        for node in nodes:
            # Skip CreateListBlock nodes that are linked to AddToList blocks
            if node.get("id") in createlist_nodes_to_remove:
                continue
                
            if node.get("block_id") == self.ADDTOLIST_BLOCK_ID:
                # Track this as an original AddToList node
                original_addtolist_node_ids.add(node.get("id"))
                original_node_id = node.get("id")
                original_node_position = node.get("metadata", {}).get("position", {})
                if original_node_position:
                    original_node_position_x = original_node_position.get("x", 0)
                    original_node_position_y = original_node_position.get("y", 0)
                else:
                    original_node_position_x = 0
                    original_node_position_y = 0
                
                # Check how many links are connected to the 'list' property of this AddToList block
                list_links_count = sum(
                    1 for link in links 
                    if link.get("sink_id") == original_node_id and link.get("sink_name") == "list"
                )
                
                # Check if there's already a prerequisite AddToList block connected to this original block
                # This prevents duplicates when the fix runs multiple times
                has_prerequisite_block = False
                for link in links:
                    if (link.get("sink_id") == original_node_id and 
                        link.get("sink_name") == "list" and
                        link.get("source_name") == "updated_list"):
                        source_node = next((n for n in nodes if n.get("id") == link.get("source_id")), None)
                        if (source_node and 
                            source_node.get("block_id") == self.ADDTOLIST_BLOCK_ID and
                            source_node.get("id") != original_node_id):  # Not self-reference
                            has_prerequisite_block = True
                            break
                
                # Check if this node is already a prerequisite block (has empty list and no incoming links to 'list')
                # This prevents treating prerequisite blocks as original blocks that need prerequisites
                is_prerequisite_block = (
                    node.get("input_default", {}).get("list") == [] and
                    node.get("input_default", {}).get("entry") is None and
                    node.get("input_default", {}).get("entries") == [] and
                    not any(link.get("sink_id") == original_node_id and link.get("sink_name") == "list" for link in links)
                )
                
                # If this is a prerequisite block, skip adding another prerequisite
                if is_prerequisite_block:
                    self.add_fix_log(f"Skipped adding prerequisite AddToList block for {original_node_id} - this is already a prerequisite block")
                # If there are already a prerequisite block exists, skip adding a new prerequisite block
                elif has_prerequisite_block:
                    self.add_fix_log(f"Skipped adding prerequisite AddToList block for {original_node_id} - already has prerequisite block exists")
                else:
                    # Generate IDs for the new nodes
                    prerequisite_node_id = self.generate_uuid()
                    
                    # Create prerequisite (empty) AddToList block with unique position
                    prerequisite_node = {
                        "id": prerequisite_node_id,
                        "block_id": self.ADDTOLIST_BLOCK_ID,
                        "input_default": {
                            "list": [],
                            "entry": None,
                            "entries": [],
                            "position": None
                        },
                        "metadata": {
                            "position": {
                                "x": original_node_position_x - 800,  # Space nodes horizontally
                                "y": original_node_position_y + 800  # Position above the original node
                            }
                        },
                        "graph_id": agent.get("id"),
                        "graph_version": 1
                    }
                    prerequisite_counter += 1
                    
                    # Create link from prerequisite to original AddToList block
                    prerequisite_link = {
                        "id": self.generate_uuid(),
                        "source_id": prerequisite_node_id,
                        "source_name": "updated_list",
                        "sink_id": original_node_id,
                        "sink_name": "list"
                    }
                    
                    # Add the prerequisite node and link
                    new_nodes.append(prerequisite_node)
                    new_links.append(prerequisite_link)
                    
                    self.add_fix_log(f"Added prerequisite AddToList block {prerequisite_node_id} before {original_node_id}")
            
            # Add the original node
            new_nodes.append(node)
        
        # Add all existing links except those marked for removal
        new_links.extend([link for link in links if link not in links_to_remove])
        
        # Check for original AddToList blocks and ensure they have self-referencing links
        for node in new_nodes:
            if (node.get("block_id") == self.ADDTOLIST_BLOCK_ID and 
                node.get("id") in original_addtolist_node_ids):  # Only original AddToList blocks
                node_id = node.get("id")
                
                # Check if this node is a prerequisite block (should not get self-referencing links)
                is_prerequisite_block = (
                    node.get("input_default", {}).get("list") == [] and
                    node.get("input_default", {}).get("entry") is None and
                    node.get("input_default", {}).get("entries") == [] and
                    not any(link.get("sink_id") == node_id and link.get("sink_name") == "list" for link in new_links)
                )
                
                # Skip self-referencing links for prerequisite blocks
                if is_prerequisite_block:
                    self.add_fix_log(f"Skipped adding self-referencing link for prerequisite AddToList block {node_id}")
                    continue
                
                # Check if this node already has a self-referencing link
                has_self_reference = any(
                    link["source_id"] == node_id and 
                    link["sink_id"] == node_id and 
                    link["source_name"] == "updated_list" and 
                    link["sink_name"] == "list"
                    for link in new_links
                )
                
                if not has_self_reference:
                    # Add self-referencing link
                    self_reference_link = {
                        "id": self.generate_uuid(),
                        "source_id": node_id,
                        "source_name": "updated_list",
                        "sink_id": node_id,
                        "sink_name": "list"
                    }
                    new_links.append(self_reference_link)
                    self.add_fix_log(f"Added self-referencing link for original AddToList block {node_id}")
        
        # Update the agent with new nodes and links
        agent["nodes"] = new_nodes
        agent["links"] = new_links
        
        return agent
    
    async def fix_addtodictionary_blocks(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix AddToDictionary blocks by removing empty CreateDictionaryBlock nodes that are linked to them.
        
        When an AddToDictionary block is found, this fixer:
        1. Checks if there's a CreateDictionaryBlock before it
        2. If CreateDictionaryBlock exists and is linked to AddToDictionary block, removes it and its link
        3. The AddToDictionary block will work with an empty dictionary as default
        
        Args:
            agent: The agent dictionary to fix
            
        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        createlist_block_id = "b924ddf4-de4f-4b56-9a85-358930dcbc91"  # CreateDictionaryBlock ID
        
        # First pass: identify CreateDictionaryBlock nodes that are linked to AddToDictionary blocks
        createlist_nodes_to_remove = set()
        links_to_remove = []
        
        for link in links:
            source_node = next((node for node in nodes if node.get("id") == link.get("source_id")), None)
            sink_node = next((node for node in nodes if node.get("id") == link.get("sink_id")), None)
            
            if (source_node and sink_node and 
                source_node.get("block_id") == createlist_block_id and 
                sink_node.get("block_id") == self.ADDTODICTIONARY_BLOCK_ID):
                createlist_nodes_to_remove.add(source_node.get("id"))
                links_to_remove.append(link)
                self.add_fix_log(f"Identified CreateDictionaryBlock {source_node.get('id')} linked to AddToDictionary block {sink_node.get('id')} for removal")
        
        # Second pass: process nodes, skipping CreateDictionaryBlock nodes that will be removed
        new_nodes = []
        for node in nodes:
            # Skip CreateDictionaryBlock nodes that are linked to AddToDictionary blocks
            if node.get("id") in createlist_nodes_to_remove:
                continue
            
            # Add the node
            new_nodes.append(node)
        
        # Remove the links that were marked for removal
        new_links = [link for link in links if link not in links_to_remove]
        
        # Update the agent with new nodes and links
        agent["nodes"] = new_nodes
        agent["links"] = new_links
        
        return agent
    
    async def fix_link_static_properties(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fix the is_static property of links based on the source block's staticOutput property.
        
        If source block's staticOutput is true, link's is_static should be true.
        If source block's staticOutput is false, link's is_static should be false.
        
        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks with their schemas
            
        Returns:
            The fixed agent dictionary
        """
        # Create a mapping of block_id to block for quick lookup
        block_map = {block.get("id"): block for block in blocks}
        
        for link in agent.get("links", []):
            # Find the source node
            source_node = next((node for node in agent.get("nodes", []) if node["id"] == link["source_id"]), None)
            if not source_node:
                continue
            
            # Get the source block
            source_block = block_map.get(source_node.get("block_id"))
            if not source_block:
                continue
            
            # Check if the source block has staticOutput property
            static_output = source_block.get("staticOutput", False)
            
            # Update the link's is_static property
            old_is_static = link.get("is_static", False)
            link["is_static"] = static_output
            
            if old_is_static != static_output:
                self.add_fix_log(f"Fixed link {link.get('id')} is_static: {old_is_static} -> {static_output} (based on source block {source_node.get('block_id')} staticOutput: {static_output})")
        
        return agent
    
    async def fix_code_execution_output(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix CodeExecutionBlock output by changing source_name from "response" to "stdout_logs" in links.
        
        Args:
            agent: The agent dictionary to fix
            
        Returns:
            The fixed agent dictionary
        """
        
        links = agent.get("links", [])
        
        for link in links:
            # Find the source node to check if it's a CodeExecutionBlock
            source_node = next((node for node in agent.get("nodes", []) if node.get("id") == link.get("source_id")), None)
            
            if (source_node and 
                source_node.get("block_id") == self.CODE_EXECUTION_BLOCK_ID and 
                link.get("source_name") == "response"):
                
                # Change source_name from "response" to "stdout_logs"
                old_source_name = link.get("source_name")
                link["source_name"] = "stdout_logs"
                self.add_fix_log(f"Fixed CodeExecutionBlock link {link.get('id')}: source_name {old_source_name} -> stdout_logs")
        
        return agent
    
    async def fix_data_sampling_sample_size(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix DataSamplingBlock by setting sample_size to 1 as default.
        If old value is set as default, just reset to 1.
        If old value is from another block, delete that link and set 1 as default.
        
        Args:
            agent: The agent dictionary to fix
            
        Returns:
            The fixed agent dictionary
        """
        
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        links_to_remove = []
        
        for node in nodes:
            if node.get("block_id") == self.DATA_SAMPLING_BLOCK_ID:
                node_id = node.get("id")
                input_default = node.get("input_default", {})
                
                # Check if there's a link to the sample_size field
                has_sample_size_link = False
                for link in links:
                    if (link.get("sink_id") == node_id and 
                        link.get("sink_name") == "sample_size"):
                        has_sample_size_link = True
                        links_to_remove.append(link)
                        self.add_fix_log(f"Removed link {link.get('id')} to DataSamplingBlock {node_id} sample_size field (will set default to 1)")
                
                # Set sample_size to 1 as default
                old_value = input_default.get("sample_size", None)
                input_default["sample_size"] = 1
                
                if has_sample_size_link:
                    self.add_fix_log(f"Fixed DataSamplingBlock {node_id} sample_size: removed link and set default to 1")
                elif old_value != 1:
                    self.add_fix_log(f"Fixed DataSamplingBlock {node_id} sample_size: {old_value} -> 1")
        
        # Remove the links that were marked for removal
        if links_to_remove:
            agent["links"] = [link for link in links if link not in links_to_remove]
        
        return agent
    
    async def fix_ai_model_parameter(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]], default_model: str = "gpt-4o") -> Dict[str, Any]:
        """
        Add default model parameter to AI blocks if missing.
        
        For nodes whose block has category "AI", this function ensures that the
        input_default has a "model" parameter. If missing, it adds the default model.
        
        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks with their schemas
            default_model: The default model to use (default "gpt-4o")
            
        Returns:
            The fixed agent dictionary
        """
        # Create a mapping of block_id to block for quick lookup
        block_map = {block.get("id"): block for block in blocks}
        
        nodes = agent.get("nodes", [])
        fixed_count = 0
        
        for node in nodes:
            block_id = node.get("block_id")
            block = block_map.get(block_id)
            
            if not block:
                continue
            
            # Check if the block has category "AI" in its categories array
            categories = block.get("categories", [])
            is_ai_block = any(
                cat.get("category") == "AI" 
                for cat in categories 
                if isinstance(cat, dict)
            )
            
            if is_ai_block:
                node_id = node.get("id")
                input_default = node.get("input_default", {})
                
                # Check if model parameter is missing
                if "model" not in input_default:
                    input_default["model"] = default_model
                    node["input_default"] = input_default
                    
                    block_name = block.get("name", "Unknown AI Block")
                    self.add_fix_log(
                        f"Added model parameter '{default_model}' to AI block node {node_id} ({block_name})"
                    )
                    fixed_count += 1
        
        if fixed_count > 0:
            logging.info(f"Added model parameter to {fixed_count} AI block nodes")
        
        return agent
    
    async def apply_all_fixes(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply all available fixes to the agent.
        
        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks with their schemas (optional)
            
        Returns:
            The fixed agent dictionary
        """
        self.fixes_applied = []
        
        # Apply fixes in order
        agent = await self.fix_agent_ids(agent)
        agent = await self.fix_double_curly_braces(agent)
        agent = await self.fix_storevalue_before_condition(agent)
        agent = await self.fix_addtolist_blocks(agent)
        agent = await self.fix_addtodictionary_blocks(agent)
        agent = await self.fix_code_execution_output(agent)
        agent = await self.fix_data_sampling_sample_size(agent)
        
        # Apply fixes that require blocks information
        if blocks:
            agent = await self.fix_ai_model_parameter(agent, blocks)
            agent = await self.fix_link_static_properties(agent, blocks)
            agent = await self.fix_data_type_mismatch(agent, blocks)
        
        logging.info(f"Applied {len(self.fixes_applied)} fixes to agent")
        for fix in self.fixes_applied:
            logging.warning(f"  - {fix}")

        return agent
    
    def get_fixes_applied(self) -> List[str]:
        """Get a list of all fixes that were applied."""
        return self.fixes_applied.copy()
    
    async def fix_data_type_mismatch(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fix data type mismatches by inserting UniversalTypeConverterBlock between incompatible connections.
        
        This function:
        1. Identifies links with type mismatches using the same logic as validate_data_type_compatibility
        2. Inserts UniversalTypeConverterBlock nodes to convert data types
        3. Rewires the connections to go through the converter block
        
        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks for reference
            
        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        
        # Create lookup dictionaries for efficiency
        block_lookup = {block["id"]: block for block in blocks}
        node_lookup = {node["id"]: node for node in nodes}
        
        def get_defined_property_type(schema, name):
            """Helper function to get property type from schema, handling nested properties."""
            if "_#_" in name:
                parent, child = name.split("_#_", 1)
                parent_schema = schema.get(parent, {})
                if "properties" in parent_schema and isinstance(parent_schema["properties"], dict):
                    return parent_schema["properties"].get(child, {}).get("type")
                else:
                    return None
            else:
                return schema.get(name, {}).get("type")
        
        def are_types_compatible(src, sink):
            """Check if two types are compatible."""
            if {src, sink} <= {"integer", "number"}:
                return True
            return src == sink
        
        def get_target_type_for_conversion(source_type, sink_type):
            """Determine the target type for conversion based on sink requirements."""
            # Map common type variations to UniversalTypeConverterBlock supported types
            type_mapping = {
                "string": "string",
                "text": "string", 
                "integer": "number",
                "number": "number",
                "float": "number",
                "boolean": "boolean",
                "bool": "boolean",
                "array": "list",
                "list": "list",
                "object": "dictionary",
                "dict": "dictionary",
                "dictionary": "dictionary"
            }
            return type_mapping.get(sink_type, sink_type)
        
        new_links = []
        nodes_to_add = []
        converter_counter = 0
        
        for link in links:
            source_node = node_lookup.get(link["source_id"])
            sink_node = node_lookup.get(link["sink_id"])
            
            if not source_node or not sink_node:
                new_links.append(link)
                continue
            
            source_block = block_lookup.get(source_node.get("block_id"))
            sink_block = block_lookup.get(sink_node.get("block_id"))
            
            if not source_block or not sink_block:
                new_links.append(link)
                continue
            
            source_outputs = source_block.get("outputSchema", {}).get("properties", {})
            sink_inputs = sink_block.get("inputSchema", {}).get("properties", {})
            
            source_type = get_defined_property_type(source_outputs, link["source_name"])
            sink_type = get_defined_property_type(sink_inputs, link["sink_name"])
            
            # Check if types are incompatible
            if source_type and sink_type and not are_types_compatible(source_type, sink_type):
                # Create UniversalTypeConverterBlock node
                converter_node_id = self.generate_uuid()
                target_type = get_target_type_for_conversion(source_type, sink_type)
                
                converter_node = {
                    "id": converter_node_id,
                    "block_id": self.UNIVERSAL_TYPE_CONVERTER_BLOCK_ID,
                    "input_default": {
                        "type": target_type
                    },
                    "metadata": {
                        "position": {
                            "x": converter_counter * 250,  # Space nodes horizontally
                            "y": 100  # Position below the source
                        }
                    },
                    "graph_id": agent.get("id"),
                    "graph_version": 1
                }
                nodes_to_add.append(converter_node)
                converter_counter += 1
                
                # Create new links: source -> converter -> sink
                source_to_converter_link = {
                    "id": self.generate_uuid(),
                    "source_id": link["source_id"],
                    "source_name": link["source_name"],
                    "sink_id": converter_node_id,
                    "sink_name": "value"
                }
                
                converter_to_sink_link = {
                    "id": self.generate_uuid(),
                    "source_id": converter_node_id,
                    "source_name": "value",
                    "sink_id": link["sink_id"],
                    "sink_name": link["sink_name"]
                }
                
                new_links.append(source_to_converter_link)
                new_links.append(converter_to_sink_link)
                
                source_block_name = source_block.get("name", "Unknown Block")
                sink_block_name = sink_block.get("name", "Unknown Block")
                self.add_fix_log(
                    f"Fixed data type mismatch: Inserted UniversalTypeConverterBlock {converter_node_id} "
                    f"between {source_block_name} ({source_type}) and {sink_block_name} ({sink_type}) "
                    f"converting to {target_type}"
                )
            else:
                # Keep the original link if types are compatible
                new_links.append(link)
        
        # Update the agent with new nodes and links
        if nodes_to_add:
            nodes.extend(nodes_to_add)
            agent["nodes"] = nodes
            agent["links"] = new_links
        
        return agent

    def clear_fixes_log(self):
        """Clear the list of applied fixes."""
        self.fixes_applied = []

class AgentValidator:
    """
    A comprehensive validator for AutoGPT agents that provides detailed error reporting
    for LLM-based fixes.
    """
    
    def __init__(self):
        self.UUID_REGEX = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}$")
        self.errors = []
    
    def is_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        return isinstance(value, str) and self.UUID_REGEX.match(value) is not None
    
    def generate_uuid(self) -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())
    
    def add_error(self, error_message: str):
        """Add an error message to the validation errors list."""
        self.errors.append(error_message)
    
    def validate_block_existence(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]]) -> bool:
        """
        Validate that all block IDs used in the agent actually exist in the blocks list.
        Returns True if all block IDs exist, False otherwise.
        """
        valid = True
        
        # Create a set of all valid block IDs for fast lookup
        valid_block_ids = {block.get("id") for block in blocks if block.get("id")}
        
        # Check each node's block_id
        for node in agent.get("nodes", []):
            block_id = node.get("block_id")
            node_id = node.get("id")
            
            if not block_id:
                self.add_error(
                    f"Node '{node_id}' is missing a 'block_id' field. "
                    f"Every node must reference a valid block."
                )
                valid = False
                continue
            
            if block_id not in valid_block_ids:
                self.add_error(
                    f"Node '{node_id}' references block_id '{block_id}' which does not exist in the available blocks. "
                    f"This block may have been deprecated, removed, or the ID is incorrect. "
                    f"Please use a valid block from the blocks library."
                )
                valid = False
        
        return valid
    
    def validate_link_node_references(self, agent: Dict[str, Any]) -> bool:
        """
        Validate that all node IDs referenced in links actually exist in the agent's nodes.
        Returns True if all link references are valid, False otherwise.
        """
        valid = True
        
        # Create a set of all valid node IDs for fast lookup
        valid_node_ids = {node.get("id") for node in agent.get("nodes", []) if node.get("id")}
        
        # Check each link's source_id and sink_id
        for link in agent.get("links", []):
            link_id = link.get("id", "Unknown")
            source_id = link.get("source_id")
            sink_id = link.get("sink_id")
            source_name = link.get("source_name", "")
            sink_name = link.get("sink_name", "")
            
            # Check source_id
            if not source_id:
                self.add_error(
                    f"Link '{link_id}' is missing a 'source_id' field. "
                    f"Every link must reference a valid source node."
                )
                valid = False
            elif source_id not in valid_node_ids:
                self.add_error(
                    f"Link '{link_id}' references source_id '{source_id}' which does not exist in the agent's nodes. "
                    f"The link from '{source_name}' cannot be established because the source node is missing."
                )
                valid = False
            
            # Check sink_id
            if not sink_id:
                self.add_error(
                    f"Link '{link_id}' is missing a 'sink_id' field. "
                    f"Every link must reference a valid sink (destination) node."
                )
                valid = False
            elif sink_id not in valid_node_ids:
                self.add_error(
                    f"Link '{link_id}' references sink_id '{sink_id}' which does not exist in the agent's nodes. "
                    f"The link to '{sink_name}' cannot be established because the destination node is missing."
                )
                valid = False
        
        return valid
    
    def validate_required_inputs(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]]) -> bool:
        """
        Validate that all required inputs are provided for each node.
        Returns True if all required inputs are satisfied, False otherwise.
        """
        valid = True
        
        for node in agent.get("nodes", []):
            block_id = node.get("block_id")
            block = next((b for b in blocks if b.get("id") == block_id), None)
            
            if not block:
                continue
            
            required_inputs = block.get("inputSchema", {}).get("required", [])
            input_defaults = node.get("input_default", {})
            node_id = node.get("id")
            
            linked_inputs = set(
                link["sink_name"]
                for link in agent.get("links", [])
                if link.get("sink_id") == node_id
            )
            
            for req_input in required_inputs:
                if req_input not in input_defaults and req_input not in linked_inputs and req_input != "credentials":
                    block_name = block.get("name", "Unknown Block")
                    self.add_error(
                        f"Node '{node_id}' (block '{block_name}' - {block_id}) is missing required input '{req_input}'. "
                        f"This input must be either provided as a default value in the node's 'input_default' field "
                        f"or connected via a link from another node's output."
                    )
                    valid = False
        
        return valid
    
    def validate_data_type_compatibility(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]]) -> bool:
        """
        Validate that linked data types are compatible between source and sink.
        Returns True if all data types are compatible, False otherwise.
        """
        valid = True
        
        for link in agent.get("links", []):
            source_node = next((node for node in agent.get("nodes", []) if node["id"] == link["source_id"]), None)
            sink_node = next((node for node in agent.get("nodes", []) if node["id"] == link["sink_id"]), None)
            
            if not source_node or not sink_node:
                continue
            
            source_block = next((b for b in blocks if b.get("id") == source_node.get("block_id")), None)
            sink_block = next((b for b in blocks if b.get("id") == sink_node.get("block_id")), None)
            
            if not source_block or not sink_block:
                continue
            
            source_outputs = source_block.get("outputSchema", {}).get("properties", {})
            sink_inputs = sink_block.get("inputSchema", {}).get("properties", {})
            
            def get_defined_property_type(schema, name):
                if "_#_" in name:
                    parent, child = name.split("_#_", 1)
                    parent_schema = schema.get(parent, {})
                    if "properties" in parent_schema and isinstance(parent_schema["properties"], dict):
                        return parent_schema["properties"].get(child, {}).get("type")
                    else:
                        return None
                else:
                    return schema.get(name, {}).get("type")
            
            source_type = get_defined_property_type(source_outputs, link["source_name"])
            sink_type = get_defined_property_type(sink_inputs, link["sink_name"])
            
            def are_types_compatible(src, sink):
                if {src, sink} <= {"integer", "number"}:
                    return True
                return src == sink

            if source_type and sink_type and not are_types_compatible(source_type, sink_type):
                source_block_name = source_block.get("name", "Unknown Block")
                sink_block_name = sink_block.get("name", "Unknown Block")
                self.add_error(
                    f"Data type mismatch in link '{link.get('id')}': "
                    f"Source '{source_block_name}' output '{link['source_name']}' outputs '{source_type}' type, "
                    f"but sink '{sink_block_name}' input '{link['sink_name']}' expects '{sink_type}' type. "
                    f"These types must match for the connection to work properly."
                )
                valid = False
        
        return valid
    
    def validate_nested_sink_links(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]]) -> bool:
        """
        Validate nested sink links (links with _#_ notation).
        Returns True if all nested links are valid, False otherwise.
        """
        valid = True
        block_input_schemas = {
            block["id"]: block.get("inputSchema", {}).get("properties", {})
            for block in blocks
        }
        block_names = {
            block["id"]: block.get("name", "Unknown Block")
            for block in blocks
        }
        
        for link in agent.get("links", []):
            sink_name = link["sink_name"]
            
            if "_#_" in sink_name:
                parent, child = sink_name.split("_#_", 1)
                
                sink_node = next((node for node in agent.get("nodes", []) if node["id"] == link["sink_id"]), None)
                if not sink_node:
                    continue
                
                block_id = sink_node.get("block_id")
                input_props = block_input_schemas.get(block_id, {})
                
                parent_schema = input_props.get(parent)
                if not parent_schema:
                    block_name = block_names.get(block_id, "Unknown Block")
                    self.add_error(
                        f"Invalid nested sink link '{sink_name}' for node '{link['sink_id']}' (block '{block_name}' - {block_id}): "
                        f"Parent property '{parent}' does not exist in the block's input schema."
                    )
                    valid = False
                    continue
                
                if not parent_schema.get("additionalProperties"):
                    if not (
                        isinstance(parent_schema, dict)
                        and "properties" in parent_schema
                        and isinstance(parent_schema["properties"], dict)
                        and child in parent_schema["properties"]
                    ):
                        block_name = block_names.get(block_id, "Unknown Block")
                        self.add_error(
                            f"Invalid nested sink link '{sink_name}' for node '{link['sink_id']}' (block '{block_name}' - {block_id}): "
                            f"Child property '{child}' does not exist in parent '{parent}' schema. "
                            f"Available properties: {list(parent_schema.get('properties', {}).keys())}"
                        )
                        valid = False
        
        return valid
    
    def validate(self, agent: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive validation of an agent against available blocks.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
            - is_valid: True if agent passes all validations, False otherwise
            - error_message: Detailed error message if validation fails, None if successful
        """
        logging.info("🔍 Validating agent...")
        self.errors = []
        
        checks = [
            ("Block existence", self.validate_block_existence(agent, blocks)),
            ("Link node references", self.validate_link_node_references(agent)),
            ("Required inputs", self.validate_required_inputs(agent, blocks)),
            ("Data type compatibility", self.validate_data_type_compatibility(agent, blocks)),
            ("Nested sink links", self.validate_nested_sink_links(agent, blocks))
        ]
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            logging.info("✅ Agent validation successful.")
            return True, None
        else:
            error_message = "Agent validation failed with the following errors:\n\n"
            for i, error in enumerate(self.errors, 1):
                error_message += f"{i}. {error}\n"
            
            error_message += "\nPlease fix these issues before the agent can be used."
            logging.error(f"❌ Agent validation failed: {error_message}")
            return False, error_message

