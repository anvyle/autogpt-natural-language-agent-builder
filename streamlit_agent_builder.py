"""
AutoGPT Agent Builder - Conversational Interface

A ChatGPT-style conversational interface for building AI agents from natural language descriptions.
Provides step-by-step guidance through goal definition, task decomposition,
and agent generation with interactive chat-like experience.
"""

import streamlit as st
import os
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# Import centralized config for secrets management
import config

from agent_builder import (
    decompose_description, 
    get_block_summaries, 
    generate_agent_json_from_subtasks,
    generate_detailed_goal,
    update_decomposition_incrementally,
    update_agent_json_incrementally,
    generate_template_modification_instructions,
)

# =============================================================================
# ENV & CONSTANTS
# =============================================================================

# Environment is set up automatically by config module
# Supports both local development (.env) and Streamlit Cloud (st.secrets)
if config.get_langchain_api_key():
    os.environ.setdefault("LANGCHAIN_TRACING", "true" if config.is_langchain_tracing_enabled() else "false")

OUTPUT_DIR = Path(f"generated_agents/{datetime.now().strftime('%Y%m%d')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AutoGPT Agent Builder",
    page_icon="🤖",
    layout="wide"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    session_vars = {
        'chat_messages': [],
        'current_step': "welcome",
        'goal': None,
        'detailed_goal': None,
        'current_decomposition': None,
        'current_decomposition_json': None,  # Raw JSON instructions
        'final_instructions': None,
        'final_instructions_json': None,  # Raw JSON instructions
        'agent_json': None,
        'clarifying_questions': None,
        'parsed_questions': [],
        'question_answers': {},
        'current_question_index': 0,
        'enhanced_goal': None,
        'waiting_for_selection': False,
        'current_options': [],
        'selected_option': None,
        'improvement_mode': False,
        'current_agent_json': None,
        'working_agent_json': None,
        'improvement_request': None,
        'chat_clarifying_questions': None,
        'chat_parsed_questions': [],
        'chat_question_answers': {},
        'updated_instructions': None,
        'updated_instructions_json': None,  # Raw JSON instructions
        'original_instructions': None,
        'original_base_instructions': None,
        'last_decomposition': None,
        'generation_counter': 0,
        'template_mode': False,
        'template_agent_json': None,
        'template_modification_instructions': None,
        'template_modification_review': None,
        'template_clarifying_questions': None,
        'template_parsed_questions': [],
        'template_question_answers': {},
        'error_message': None,  # To store error messages for UI display
    }
    
    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data
def load_blocks() -> Tuple[Optional[List], Optional[Dict]]:
    """Load and cache block summaries and blocks."""
    try:
        block_summaries, blocks = asyncio.run(get_block_summaries())
        return block_summaries, blocks
    except Exception as e:
        st.error(f"Failed to load blocks: {e}")
        return None, None

def add_message(content: str, is_user: bool = False, message_type: str = "text"):
    """Add a message to the chat history."""
    st.session_state.chat_messages.append({
        "content": content,
        "is_user": is_user,
        "type": message_type,
        "timestamp": datetime.now()
    })

def add_system_message(content: str):
    """Add a system message to the chat."""
    add_message(content, is_user=False, message_type="system")

def add_user_message(content: str):
    """Add a user message to the chat."""
    add_message(content, is_user=True, message_type="text")

def add_assistant_message(content: str):
    """Add an assistant message to the chat."""
    add_message(content, is_user=False, message_type="text")

def add_instructions_message(content: str):
    """Add an instructions message that displays content as plain text without Markdown interpretation."""
    add_message(content, is_user=False, message_type="instructions")

def add_options_message(options: List[str], message: str = "Please select an option:"):
    """Add a message with selectable options."""
    add_message({
        "text": message,
        "options": options
    }, is_user=False, message_type="options")

def add_agent_results_message(agent_json: dict, filename: str, is_updated: bool = False):
    """Add a message with agent results including metrics and download button."""
    add_message({
        "agent_json": agent_json,
        "filename": filename,
        "is_updated": is_updated
    }, is_user=False, message_type="agent_results")

def add_template_upload_message():
    """Add a message with file upload for template agent."""
    add_message({
        "text": "📁 Please upload your template agent.json file:"
    }, is_user=False, message_type="template_upload")

def parse_clarifying_questions(questions_text: str) -> List[Dict]:
    """Parse clarifying questions from the LLM response format."""
    parsed_questions = []
    
    try:
        if isinstance(questions_text, str):
            content = questions_text.strip("```json").strip("```").strip()
            data = json.loads(content)
        else:
            data = questions_text
            
        if isinstance(data, dict) and data.get("type") == "clarifying_questions":
            return data.get("questions", [])
    except (json.JSONDecodeError, AttributeError):
        pass
    
    if not questions_text or "❓ Clarifying Questions:" not in questions_text:
        return parsed_questions
    
    lines = questions_text.strip().split('\n')
    current_question = None
    current_keyword = None
    current_example = None
    
    for line in lines:
        line = line.strip()
        
        if not line or line == "❓ Clarifying Questions:":
            continue
        
        if line.startswith("- "):
            if current_question:
                parsed_questions.append({
                    'question': current_question,
                    'keyword': current_keyword,
                    'example': current_example
                })
            
            current_question = line[2:]  # Remove "- " prefix
            current_keyword = None
            current_example = None
            
        elif ":" in line and "e.g.," in line:
            parts = line.split(":", 1)
            if len(parts) >= 2:
                current_keyword = parts[0].strip()
                example_part = parts[1].strip()
                if "e.g.," in example_part:
                    current_example = example_part.split("e.g.,")[1].strip().strip('"')
                else:
                    current_example = example_part
    
    if current_question:
        parsed_questions.append({
            'question': current_question,
            'keyword': current_keyword,
            'example': current_example
        })
    
    return parsed_questions

def reset_chat():
    """Reset the chat for a new agent generation."""
    st.session_state.chat_messages = []
    st.session_state.current_step = "welcome"
    st.session_state.goal = None
    st.session_state.detailed_goal = None
    st.session_state.current_decomposition = None
    st.session_state.current_decomposition_json = None
    st.session_state.final_instructions = None
    st.session_state.final_instructions_json = None
    st.session_state.agent_json = None
    st.session_state.clarifying_questions = None
    st.session_state.parsed_questions = []
    st.session_state.question_answers = {}
    st.session_state.current_question_index = 0
    st.session_state.enhanced_goal = None
    st.session_state.waiting_for_selection = False
    st.session_state.current_options = []
    st.session_state.selected_option = None
    st.session_state.improvement_mode = False
    st.session_state.current_agent_json = None
    st.session_state.working_agent_json = None
    st.session_state.improvement_request = None
    st.session_state.chat_clarifying_questions = None
    st.session_state.chat_parsed_questions = []
    st.session_state.chat_question_answers = {}
    st.session_state.updated_instructions = None
    st.session_state.updated_instructions_json = None
    st.session_state.original_instructions = None
    st.session_state.original_base_instructions = None
    st.session_state.last_decomposition = None
    st.session_state.generation_counter = 0
    st.session_state.template_mode = False
    st.session_state.template_agent_json = None
    st.session_state.template_modification_instructions = None
    st.session_state.template_modification_review = None
    st.session_state.template_clarifying_questions = None
    st.session_state.template_parsed_questions = []
    st.session_state.template_question_answers = {}
    st.session_state.error_message = None

# =============================================================================
# STAGE-SPECIFIC UI RENDERING
# =============================================================================

def render_error_message():
    """Render error message if present in session state."""
    if st.session_state.error_message:
        st.error(f"❌ {st.session_state.error_message}")
        # Clear error after displaying
        st.session_state.error_message = None

def render_welcome_stage():
    """Render the welcome stage."""
    render_error_message()
    st.title("🤖 AutoGPT Agent Builder")
    st.markdown("Build AI agents through natural conversation")
    
    st.info("🤖 **Welcome to AutoGPT Agent Builder!**")
    st.write("I'm here to help you create AI agents through natural conversation. What would you like to do?")
    
    # Display options
    options = [
        "Create New Agent",
        "Modify Template Agent"
    ]
    
    cols = st.columns(len(options))
    for i, option in enumerate(options):
        with cols[i]:
            if st.button(option, key=f"welcome_option_{i}", use_container_width=True):
                st.session_state.selected_option = option
                handle_option_selection(option)
                st.rerun()

def render_goal_input_stage():
    """Render the goal input stage."""
    render_error_message()
    st.title("🎯 Define Your Goal")
    st.markdown("**Step 1: Define Your Goal**")
    
    st.write("Please describe what you want your agent to do. Be as specific as possible about the task, inputs, outputs, and any special requirements.")
    
    # Show current goal if available
    if st.session_state.goal:
        st.info(f"**Current Goal:** {st.session_state.goal}")
    
    # Input area
    render_input_area()

def render_goal_refinement_stage():
    """Render the goal refinement stage."""
    render_error_message()
    st.title("🎯 Review Your Goal")
    st.markdown("**Step 1: Review Your Goal**")
    
    if st.session_state.detailed_goal:
        st.write("✅ I've generated a detailed goal based on your description:")
        st.info(f"**Detailed Goal:**\n{st.session_state.detailed_goal}")
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Use as is", key="use_goal", use_container_width=True):
                st.session_state.selected_option = "Use as is"
                handle_option_selection("Use as is")
                st.rerun()
        with col2:
            if st.button("✏️ Edit goal", key="edit_goal", use_container_width=True):
                st.session_state.selected_option = "Edit goal"
                handle_option_selection("Edit goal")
                st.rerun()

def render_goal_suggestion_stage():
    """Render the goal suggestion stage."""
    render_error_message()
    st.title("🎯 Goal Suggestion")
    st.markdown("**Step 1: Goal Suggestion**")
    
    if hasattr(st.session_state, 'last_decomposition') and st.session_state.last_decomposition:
        decomposition = st.session_state.last_decomposition
        message = decomposition.get("message", "")
        reason = decomposition.get("reason", "")
        suggested_goal = decomposition.get("suggested_goal", "")
        
        st.warning(f"**{message}**")
        if reason:
            st.write(f"**Reason:** {reason}")
        if suggested_goal:
            st.info(f"**Suggested Alternative:** {suggested_goal}")
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Use Suggested Goal", key="use_suggested", use_container_width=True):
                st.session_state.selected_option = "Use Suggested Goal"
                handle_option_selection("Use Suggested Goal")
                st.rerun()
        with col2:
            if st.button("🔄 Try Different Goal", key="try_different", use_container_width=True):
                st.session_state.selected_option = "Try Different Goal"
                handle_option_selection("Try Different Goal")
                st.rerun()

def render_clarification_stage():
    """Render the clarification stage."""
    render_error_message()
    st.title("❓ Clarifying Questions")
    st.markdown("**Step 2: Additional Information**")
    
    # Determine which questions to show based on mode
    if st.session_state.improvement_mode:
        questions = st.session_state.chat_parsed_questions
        st.write("I need more information to make your improvement:")
    elif st.session_state.template_mode:
        questions = st.session_state.template_parsed_questions
        st.write("I need more information to modify your template effectively:")
    else:
        questions = st.session_state.parsed_questions
        st.write("I need some additional information to create your workflow effectively:")
    
    if questions:
        st.write("**Clarifying Questions:**")
        for i, question in enumerate(questions):
            with st.expander(f"Question {i+1}: {question.get('question', '')[:50]}..."):
                st.write(f"**Question:** {question.get('question', '')}")
                if question.get('keyword'):
                    st.write(f"**Category:** {question['keyword']}")
                if question.get('example'):
                    st.write(f"**Example:** {question['example']}")
                
                if st.button(f"Answer Question {i+1}", key=f"answer_q_{i}"):
                    st.session_state.selected_option = f"Question {i+1}: {question['question'][:50]}..."
                    handle_option_selection(st.session_state.selected_option)
                    st.rerun()

def render_answering_question_stage():
    """Render the answering question stage."""
    render_error_message()
    st.title("❓ Answer Question")
    st.markdown("**Step 2: Answer Question**")
    
    # Determine which questions to show based on mode
    if st.session_state.improvement_mode:
        questions = st.session_state.chat_parsed_questions
    elif st.session_state.template_mode:
        questions = st.session_state.template_parsed_questions
    else:
        questions = st.session_state.parsed_questions
    
    current_index = st.session_state.current_question_index
    if current_index < len(questions):
        question_data = questions[current_index]
        
        st.write(f"**Question {current_index + 1}:** {question_data['question']}")
        if question_data.get('keyword'):
            st.write(f"**Category:** {question_data['keyword']}")
        if question_data.get('example'):
            st.write(f"**Example:** {question_data['example']}")
        
        st.write("Please provide your answer:")
        render_input_area()

def render_decomposition_review_stage():
    """Render the decomposition review stage."""
    render_error_message()
    st.title("📋 Review Instructions")
    
    # Determine if we're in improvement mode or initial creation
    if st.session_state.improvement_mode:
        st.markdown("**Step 3: Review Updated Instructions**")
        instructions_to_show = st.session_state.updated_instructions
        if instructions_to_show:
            st.write("✅ I've updated the instructions based on your request:")
            st.text_area("Updated Instructions:", instructions_to_show, height=300, disabled=True)
            
            # Options for improvement mode
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🚀 Generate Updated Agent", key=f"generate_updated {st.session_state.generation_counter + 1}", use_container_width=True):
                    st.session_state.selected_option = f"Generate Updated Agent"
                    handle_option_selection(st.session_state.selected_option)
                    st.rerun()
            with col2:
                if st.button("✏️ Edit instructions", key="edit_instructions", use_container_width=True):
                    st.session_state.selected_option = "Edit instructions"
                    handle_option_selection("Edit instructions")
                    st.rerun()
        else:
            st.error("No updated instructions available. Please try again.")
    else:
        st.markdown("**Step 3: Review Instructions**")
        if st.session_state.current_decomposition:
            st.write("✅ I've generated step-by-step instructions for your goal:")
            st.text_area("Instructions:", st.session_state.current_decomposition, height=300, disabled=True)
            
            # Options for initial creation
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Looks good", key="looks_good", use_container_width=True):
                    st.session_state.selected_option = "Looks good"
                    handle_option_selection("Looks good")
                    st.rerun()
            with col2:
                if st.button("✏️ Edit instructions", key="edit_instructions", use_container_width=True):
                    st.session_state.selected_option = "Edit instructions"
                    handle_option_selection("Edit instructions")
                    st.rerun()
        else:
            st.error("No instructions available. Please try again.")

def render_final_stage():
    """Render the final stage before generation."""
    render_error_message()
    st.title("🚀 Ready to Generate")
    
    # Determine if we're in improvement mode or initial creation
    if st.session_state.improvement_mode:
        st.markdown("**Step 4: Final Review - Updated Agent**")
        instructions_to_show = st.session_state.updated_instructions
        if instructions_to_show:
            st.write("✅ Updated instructions finalized! Ready to generate your updated agent.")
            st.text_area("Final Updated Instructions:", instructions_to_show, height=300, disabled=True)
            
            # Options for improvement mode
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🚀 Generate Updated Agent", key="generate_updated_final", use_container_width=True):
                    st.session_state.selected_option = f"Generate Updated Agent"
                    handle_option_selection(st.session_state.selected_option)
                    st.rerun()
            with col2:
                if st.button("✏️ Edit instructions", key="edit_final", use_container_width=True):
                    st.session_state.selected_option = "Edit instructions"
                    handle_option_selection("Edit instructions")
                    st.rerun()
        else:
            st.error("No updated instructions available. Please try again.")
    else:
        st.markdown("**Step 4: Final Review**")
        if st.session_state.final_instructions:
            st.write("✅ Instructions finalized! Ready to generate your agent.")
            st.text_area("Final Instructions:", st.session_state.final_instructions, height=300, disabled=True)
            
            # Options for initial creation
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Generate Agent", key="generate_agent", use_container_width=True):
                    st.session_state.selected_option = "Generate Agent"
                    handle_option_selection("Generate Agent")
                    st.rerun()
            with col2:
                if st.button("✏️ Edit instructions", key="edit_final", use_container_width=True):
                    st.session_state.selected_option = "Edit instructions"
                    handle_option_selection("Edit instructions")
                    st.rerun()
        else:
            st.error("No final instructions available. Please try again.")

def render_agent_results_stage():
    """Render the agent results stage."""
    render_error_message()
    # Determine if this is an updated agent or initial agent
    if st.session_state.generation_counter > 0:
        st.title(f"🎉 Updated Agent #{st.session_state.generation_counter} Generated!")
        st.markdown("**Step 5: Your Updated Agent is Ready**")
    else:
        st.title("🎉 Agent Generated!")
        st.markdown("**Step 5: Your Agent is Ready**")
    
    if st.session_state.agent_json:
        agent_json = st.session_state.agent_json
        filename = re.sub(r'[^a-zA-Z0-9]+', '_', agent_json.get("name", "agent")).strip('_')[:50]
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Name", agent_json.get("name", "N/A"))
        with col2:
            st.metric("Nodes", len(agent_json.get("nodes", [])))
        with col3:
            st.metric("Links", len(agent_json.get("links", [])))
        
        # Download button with appropriate label
        if st.session_state.generation_counter > 0:
            download_label = f"📥 Download Updated Agent JSON #{st.session_state.generation_counter}"
        else:
            download_label = "📥 Download Agent JSON"
            
        st.download_button(
            label=download_label,
            data=json.dumps(agent_json, indent=2),
            file_name=f"{filename}.json",
            mime="application/json",
            key="download_agent"
        )
        
        if st.session_state.generation_counter > 0:
            st.success(f"Your updated agent #{st.session_state.generation_counter} is ready! You can download it above or start a new agent.")
        else:
            st.success("Your agent is ready! You can download it above or start a new agent.")
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 Start New Agent", key="start_new", use_container_width=True):
                st.session_state.selected_option = "Start New Agent"
                handle_option_selection("Start New Agent")
                st.rerun()
        with col2:
            if st.button("🔧 Improve This Agent", key="improve_agent", use_container_width=True):
                st.session_state.selected_option = "Improve This Agent"
                handle_option_selection("Improve This Agent")
                st.rerun()

def render_agent_chat_stage():
    """Render the agent improvement chat stage."""
    render_error_message()
    st.title("💬 Agent Improvement")
    st.markdown("**Agent Improvement Mode**")
    
    st.write("I'm here to help you improve your existing agent. Please describe what changes you'd like to make to your agent.")
    
    # Show current agent info if available
    if st.session_state.agent_json:
        agent_json = st.session_state.agent_json
        with st.expander("Current Agent Info"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Name", agent_json.get("name", "N/A"))
            with col2:
                st.metric("Nodes", len(agent_json.get("nodes", [])))
            with col3:
                st.metric("Links", len(agent_json.get("links", [])))
    
    render_input_area()

def render_template_upload_stage():
    """Render the template upload stage."""
    render_error_message()
    st.title("📁 Template Agent Upload")
    st.markdown("**Template Agent Modification Mode**")
    
    st.write("I'll help you modify an existing agent template. First, please upload your template agent.json file.")
    
    render_template_upload()

def render_template_instructions_stage():
    """Render the template instructions stage."""
    render_error_message()
    st.title("📝 Template Modifications")
    st.markdown("**Step 2: Describe Modifications**")
    
    if st.session_state.template_agent_json:
        agent_json = st.session_state.template_agent_json
        st.info(f"**Template Agent:** {agent_json.get('name', 'Unnamed')} (Nodes: {len(agent_json.get('nodes', []))}, Links: {len(agent_json.get('links', []))})")
    
    st.write("Now, please describe how you want to modify this template agent. Be specific about what changes you want to make (e.g., add new steps, modify existing functionality, change outputs, etc.).")
    
    render_input_area()

def render_template_modification_review_stage():
    """Render the template modification review stage."""
    render_error_message()
    st.title("📋 Review Template Modifications")
    st.markdown("**Step 3: Review Modifications**")
    
    if st.session_state.updated_instructions:
        st.write("✅ I've generated modified instructions based on your template and request:")
        st.text_area("Modified Instructions:", st.session_state.updated_instructions, height=300, disabled=True)
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Generate Modified Agent", key="generate_modified", use_container_width=True):
                st.session_state.selected_option = "Generate Modified Agent"
                handle_option_selection("Generate Modified Agent")
                st.rerun()
        with col2:
            if st.button("✏️ Edit modifications", key="edit_modifications", use_container_width=True):
                st.session_state.selected_option = "Edit modifications"
                handle_option_selection("Edit modifications")
                st.rerun()

def render_current_stage():
    """Render the current stage based on current_step."""
    current_step = st.session_state.current_step
    
    if current_step == "welcome":
        render_welcome_stage()
    elif current_step == "goal_input":
        render_goal_input_stage()
    elif current_step == "goal_refinement":
        render_goal_refinement_stage()
    elif current_step == "goal_suggestion":
        render_goal_suggestion_stage()
    elif current_step == "clarification":
        render_clarification_stage()
    elif current_step == "answering_question":
        render_answering_question_stage()
    elif current_step == "decomposition_review":
        render_decomposition_review_stage()
    elif current_step == "final":
        render_final_stage()
    elif current_step == "agent_results":
        render_agent_results_stage()
    elif current_step == "agent_chat":
        render_agent_chat_stage()
    elif current_step == "template_upload":
        render_template_upload_stage()
    elif current_step == "template_instructions":
        render_template_instructions_stage()
    elif current_step == "template_modification_review":
        render_template_modification_review_stage()
    else:
        st.error(f"Unknown stage: {current_step}")

# =============================================================================
# CHAT RENDERING (LEGACY - KEPT FOR COMPATIBILITY)
# =============================================================================

def render_chat_interface():
    """Render the main chat interface (legacy function)."""
    # This function is kept for compatibility but now redirects to stage-based rendering
    render_current_stage()

def render_options(options: List[str], message_index: int):
    """Render selectable options."""
    cols = st.columns(len(options))
    for i, option in enumerate(options):
        with cols[i]:
            if st.button(option, key=f"option_{message_index}_{i}"):
                st.session_state.selected_option = option
                st.session_state.waiting_for_selection = False
                handle_option_selection(option)
                st.rerun()

def render_agent_results(content: dict, message_index: int):
    """Render agent results with metrics and download button."""
    agent_json = content["agent_json"]
    filename = content["filename"]
    is_updated = content["is_updated"]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Name", agent_json.get("name", "N/A"))
    with col2:
        st.metric("Nodes", len(agent_json.get("nodes", [])))
    with col3:
        st.metric("Links", len(agent_json.get("links", [])))
    
    # Download button
    if is_updated:
        button_label = f"📥 Download Updated Agent JSON #{st.session_state.generation_counter}"
    else:
        button_label = "📥 Download Agent JSON"
    st.download_button(
        label=button_label,
        data=json.dumps(agent_json, indent=2),
        file_name=f"{filename}.json",
        mime="application/json",
        key=f"download_{message_index}"
    )

def render_template_upload():
    """Render template agent file upload interface."""
    uploaded_file = st.file_uploader(
        "Choose an agent.json file",
        type=['json'],
        key="template_uploader",
        help="Upload a valid agent.json file to use as a template"
    )
    
    if uploaded_file is not None:
        try:
            # Read and parse the uploaded file
            content = uploaded_file.read().decode('utf-8')
            agent_json = json.loads(content)
            
            # Validate the agent JSON
            is_valid, error = validate_template_agent(agent_json)
            
            if is_valid:
                st.success("✅ Template agent loaded successfully!")
                
                # Display agent info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Name", agent_json.get("name", "N/A"))
                with col2:
                    st.metric("Nodes", len(agent_json.get("nodes", [])))
                with col3:
                    st.metric("Links", len(agent_json.get("links", [])))
                
                # Store the template agent
                st.session_state.template_agent_json = agent_json
                st.session_state.working_agent_json = agent_json
                
                # Proceed to modification instructions
                if st.button("Continue with Template", key="continue_template"):
                    handle_template_uploaded(agent_json)
                    st.rerun()
            else:
                st.error(f"❌ Invalid agent JSON: {error}")
                st.text_area("File content:", content, height=200, disabled=True)
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {e}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

def validate_template_agent(agent_json: dict) -> Tuple[bool, str]:
    """Validate uploaded template agent JSON."""
    try:
        # Check required fields
        required_fields = ['id', 'name', 'description', 'nodes', 'links']
        for field in required_fields:
            if field not in agent_json:
                return False, f"Missing required field: {field}"
        
        # Check nodes structure
        if not isinstance(agent_json['nodes'], list):
            return False, "Nodes must be a list"
        
        # Check links structure
        if not isinstance(agent_json['links'], list):
            return False, "Links must be a list"
        
        # Basic validation - ensure each node has required fields
        for i, node in enumerate(agent_json['nodes']):
            if not isinstance(node, dict):
                return False, f"Node {i} must be a dictionary"
            if 'id' not in node or 'block_id' not in node:
                return False, f"Node {i} missing required fields (id, block_id)"
        
        # Basic validation - ensure each link has required fields
        for i, link in enumerate(agent_json['links']):
            if not isinstance(link, dict):
                return False, f"Link {i} must be a dictionary"
            required_link_fields = ['id', 'source_id', 'source_name', 'sink_id', 'sink_name']
            for field in required_link_fields:
                if field not in link:
                    return False, f"Link {i} missing required field: {field}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"

def render_input_area():
    """Render the input area for user messages."""
    if st.session_state.current_step in ["goal_input", "answering_question", "agent_chat", "template_instructions"]:
        if st.session_state.current_step == "goal_input":
            placeholder = "Describe your goal (e.g., 'Create an agent that sends daily weather reports')"
        elif st.session_state.current_step == "answering_question":
            placeholder = "Provide your answer to the question above"
        elif st.session_state.current_step == "agent_chat":
            placeholder = "Describe what you want to improve (e.g., 'Add error handling for email sending')"
        elif st.session_state.current_step == "template_instructions":
            placeholder = "Describe how you want to modify the template agent (e.g., 'Add email notifications after each step')"
        
        user_input = st.chat_input(placeholder)
        
        if user_input:
            add_user_message(user_input)
            handle_user_input(user_input)
            st.rerun()

# =============================================================================
# STEP HANDLERS
# =============================================================================

def handle_user_input(user_input: str):
    """Handle user input based on current step."""
    if st.session_state.current_step == "goal_input":
        handle_goal_input(user_input)
    elif st.session_state.current_step == "answering_question":
        if st.session_state.improvement_mode:
            handle_improvement_question_answer(user_input)
        elif st.session_state.template_mode:
            handle_template_question_answer(user_input)
        else:
            handle_question_answer(user_input)
    elif st.session_state.current_step == "agent_chat":
        handle_improvement_request(user_input)
    elif st.session_state.current_step == "template_instructions":
        handle_template_modification_request(user_input)

def handle_option_selection(option: str):
    """Handle option selection based on current step."""
    if st.session_state.current_step == "welcome":
        if option == "Create New Agent":
            start_new_agent_creation()
        elif option == "Modify Template Agent":
            start_template_modification()
    elif st.session_state.current_step == "goal_refinement":
        if option == "Use as is":
            proceed_to_decomposition()
        elif option == "Edit goal":
            st.session_state.current_step = "goal_input"
            st.rerun()
    elif st.session_state.current_step == "goal_suggestion":
        if option == "Use Suggested Goal":
            # Use the suggested goal from the last decomposition response
            if hasattr(st.session_state, 'last_decomposition') and st.session_state.last_decomposition:
                if st.session_state.last_decomposition.get("type") in ["unachievable_goal", "vague_goal"]:
                    suggested_goal = st.session_state.last_decomposition.get("suggested_goal", "")
                    if suggested_goal:
                        st.session_state.goal = suggested_goal
                        st.session_state.detailed_goal = suggested_goal
                        proceed_to_decomposition()
                    else:
                        st.error("❌ No suggested goal available. Please try a different goal.")
                        st.session_state.current_step = "goal_input"
                        st.rerun()
            else:
                st.error("❌ No suggested goal available. Please try a different goal.")
                st.session_state.current_step = "goal_input"
                st.rerun()
        elif option == "Try Different Goal":
            st.session_state.current_step = "goal_input"
            st.rerun()
    elif st.session_state.current_step == "clarification":
        if st.session_state.improvement_mode:
            handle_improvement_clarification_selection(option)
        elif st.session_state.template_mode:
            handle_template_clarification_selection(option)
        else:
            handle_clarification_selection(option)
    elif st.session_state.current_step == "decomposition_review":
        if option == "Looks good":
            proceed_to_generation()
        elif option == "Edit instructions":
            st.session_state.current_step = "goal_input"
            st.rerun()
        elif option == "Generate Updated Agent":
            generate_updated_agent()
    elif st.session_state.current_step == "final":
        if option == "Generate Agent":
            generate_agent()
        elif option == "Edit instructions":
            st.session_state.current_step = "decomposition_review"
            st.rerun()
        elif option == "Generate Updated Agent":
            generate_updated_agent()
    elif st.session_state.current_step == "agent_chat":
        if option.startswith("Generate Updated Agent"):
            generate_updated_agent()
        elif option == "Edit instructions":
            st.session_state.current_step = "decomposition_review"
            st.rerun()
        elif option == "Start New Agent":
            reset_chat()
            start_new_agent_creation()
        elif option == "Try Different Improvement":
            st.session_state.current_step = "agent_chat"
            st.rerun()
        elif option == "Generate Modified Agent":
            generate_modified_agent_from_template()
        elif option == "Edit modifications":
            st.session_state.current_step = "template_instructions"
            st.rerun()
    elif st.session_state.current_step == "template_modification_review":
        if option == "Generate Modified Agent":
            generate_modified_agent_from_template()
        elif option == "Edit modifications":
            st.session_state.current_step = "template_instructions"
            st.rerun()
    elif st.session_state.current_step == "agent_results":
        if option == "Start New Agent":
            reset_chat()
            start_new_agent_creation()
        elif option == "Improve This Agent":
            st.session_state.current_step = "agent_chat"
            st.session_state.improvement_mode = True
            # Set the working agent JSON to the current agent JSON (will be updated with each improvement)
            st.session_state.working_agent_json = st.session_state.agent_json
            st.rerun()

def start_new_agent_creation():
    """Start the new agent creation process."""
    st.session_state.current_step = "goal_input"

def start_agent_improvement():
    """Start the agent improvement process."""
    st.session_state.current_step = "agent_chat"
    st.session_state.improvement_mode = True
    st.session_state.working_agent_json = st.session_state.agent_json

def start_template_modification():
    """Start the template agent modification process."""
    st.session_state.current_step = "template_upload"
    st.session_state.template_mode = True

def handle_template_uploaded(agent_json: dict):
    """Handle successfully uploaded template agent."""
    st.session_state.template_agent_json = agent_json
    st.session_state.working_agent_json = agent_json
    st.session_state.current_step = "template_instructions"
    st.rerun()

def handle_template_modification_request(modification_request: str):
    """Handle template modification request."""
    st.session_state.template_modification_instructions = modification_request
    st.session_state.error_message = None  # Clear previous errors
    
    with st.spinner("Processing template modification request..."):
        try:
            # Generate instructions based on the template and modification request
            updated_instructions = asyncio.run(
                generate_template_modification_instructions(
                    st.session_state.template_agent_json,
                    modification_request,
                    block_summaries
                )
            )
            
            if not updated_instructions:
                st.session_state.error_message = "Failed to generate modification instructions"
                st.rerun()
                return
            
            # Check if updated_instructions contains clarifying questions (new JSON format)
            if isinstance(updated_instructions, dict) and updated_instructions.get("type") == "clarifying_questions":
                st.session_state.template_clarifying_questions = updated_instructions
                st.session_state.template_parsed_questions = updated_instructions.get("questions", [])
                # Switch step so option clicks are handled by clarification flow
                st.session_state.current_step = "clarification"
                st.rerun()
                
            # Handle unachievable modification (new JSON format)
            elif isinstance(updated_instructions, dict) and updated_instructions.get("type") == "unachievable_goal":
                st.session_state.last_decomposition = updated_instructions
                st.session_state.current_step = "goal_suggestion"
                st.rerun()
                
            # Handle vague modification (new JSON format)
            elif isinstance(updated_instructions, dict) and updated_instructions.get("type") == "vague_goal":
                st.session_state.last_decomposition = updated_instructions
                st.session_state.current_step = "goal_suggestion"
                st.rerun()
                
            # Check for old text format
            elif isinstance(updated_instructions, str) and "❓ Clarifying Questions:" in updated_instructions:
                st.session_state.template_clarifying_questions = updated_instructions
                st.session_state.template_parsed_questions = parse_clarifying_questions(updated_instructions)
                st.session_state.current_step = "clarification"
                st.rerun()
                
            # Handle updated instructions (new JSON format)
            elif isinstance(updated_instructions, dict) and updated_instructions.get("type") == "instructions":
                instructions_text = "**Modified Step-by-Step Instructions:**\n"
                for step in updated_instructions.get("steps", []):
                    instructions_text += f"{step.get('step_number', '')}. {step.get('description', '')}\n"
                    if step.get('inputs'):
                        instructions_text += "   Inputs:\n"
                        for input_item in step['inputs']:
                            instructions_text += f"   - {input_item.get('name', '')}: {input_item.get('value', '')}\n"
                    if step.get('outputs'):
                        instructions_text += "   Outputs:\n"
                        for output_item in step['outputs']:
                            instructions_text += f"   - {output_item.get('name', '')}: {output_item.get('description', '')}\n"
                    instructions_text += "\n"
                
                # Store both formatted text for UI and raw JSON for agent generation
                st.session_state.updated_instructions = instructions_text
                st.session_state.updated_instructions_json = updated_instructions
                st.session_state.current_step = "template_modification_review"
                st.rerun()
                
            else:
                st.session_state.updated_instructions = ""
                st.session_state.updated_instructions_json = updated_instructions
                st.session_state.current_step = "template_modification_review"
                st.rerun()
                
        except Exception as e:
            st.session_state.error_message = f"Error processing template modification request: {str(e)}"
            st.rerun()

def handle_goal_input(goal: str):
    """Handle goal input and generate detailed goal."""
    st.session_state.goal = goal
    st.session_state.error_message = None  # Clear previous errors
    
    with st.spinner("Generating detailed goal..."):
        try:
            detailed_goal = asyncio.run(generate_detailed_goal(goal, block_summaries))
            
            if detailed_goal:
                st.session_state.detailed_goal = detailed_goal
                st.session_state.current_step = "goal_refinement"
                st.rerun()
            else:
                st.session_state.error_message = "I couldn't generate a detailed goal. Please try again with a more specific description."
                st.rerun()
        except Exception as e:
            st.session_state.error_message = f"Error generating detailed goal: {str(e)}"
            st.rerun()

def proceed_to_decomposition():
    """Proceed to task decomposition."""
    goal_to_use = st.session_state.enhanced_goal or st.session_state.detailed_goal or st.session_state.goal
    st.session_state.error_message = None  # Clear previous errors
    
    with st.spinner("Generating step-by-step instructions..."):
        try:
            decomposition = asyncio.run(decompose_description(goal_to_use, block_summaries))
            
            if isinstance(decomposition, dict) and decomposition.get("type") == "clarifying_questions":
                st.session_state.clarifying_questions = decomposition
                st.session_state.parsed_questions = decomposition.get("questions", [])
                st.session_state.current_step = "clarification"
                st.rerun()
                
            elif isinstance(decomposition, dict) and decomposition.get("type") == "unachievable_goal":
                st.session_state.last_decomposition = decomposition
                st.session_state.current_step = "goal_suggestion"
                st.rerun()
                
            elif isinstance(decomposition, dict) and decomposition.get("type") == "vague_goal":
                st.session_state.last_decomposition = decomposition
                st.session_state.current_step = "goal_suggestion"
                st.rerun()
                
            elif isinstance(decomposition, str) and "❓ Clarifying Questions:" in decomposition:
                st.session_state.clarifying_questions = decomposition
                st.session_state.parsed_questions = parse_clarifying_questions(decomposition)
                st.session_state.current_step = "clarification"
                st.rerun()
                
            elif isinstance(decomposition, dict) and decomposition.get("type") == "instructions":
                instructions_text = "**Step-by-Step Instructions:**\n"
                for step in decomposition.get("steps", []):
                    instructions_text += f"{step.get('step_number', '')}. {step.get('description', '')}\n"
                    if step.get('inputs'):
                        instructions_text += "   Inputs:\n"
                        for input_item in step['inputs']:
                            instructions_text += f"   - {input_item.get('name', '')}: {input_item.get('value', '')}\n"
                    if step.get('outputs'):
                        instructions_text += "   Outputs:\n"
                        for output_item in step['outputs']:
                            instructions_text += f"   - {output_item.get('name', '')}: {output_item.get('description', '')}\n"
                    instructions_text += "\n"
                
                # Store both formatted text for UI and raw JSON for agent generation
                st.session_state.current_decomposition = instructions_text
                st.session_state.current_decomposition_json = decomposition
                st.session_state.current_step = "decomposition_review"
                st.rerun()
                
            else:
                st.session_state.current_decomposition = decomposition
                st.session_state.current_step = "decomposition_review"
                st.rerun()
        except Exception as e:
            st.session_state.error_message = f"Error generating instructions: {str(e)}"
            st.rerun()

def handle_clarification_selection(option: str):
    """Handle clarification question selection."""
    try:
        question_num = int(option.split("Question ")[1].split(":")[0])
        question_index = question_num - 1
    except (ValueError, IndexError):
        st.session_state.error_message = "Error: Could not parse question number. Please try again."
        st.rerun()
        return
    
    if question_index < 0 or question_index >= len(st.session_state.parsed_questions):
        st.session_state.error_message = "Error: Invalid question number. Please try again."
        st.rerun()
        return
    
    st.session_state.current_question_index = question_index
    st.session_state.current_step = "answering_question"
    st.rerun()

def handle_improvement_clarification_selection(option: str):
    """Handle clarification question selection in improvement mode."""
    try:
        question_num = int(option.split("Question ")[1].split(":")[0])
        question_index = question_num - 1
    except (ValueError, IndexError):
        st.session_state.error_message = "Error: Could not parse question number. Please try again."
        st.rerun()
        return
    
    if question_index < 0 or question_index >= len(st.session_state.chat_parsed_questions):
        st.session_state.error_message = "Error: Invalid question number. Please try again."
        st.rerun()
        return
    
    st.session_state.current_question_index = question_index
    st.session_state.current_step = "answering_question"
    st.rerun()

def handle_template_clarification_selection(option: str):
    """Handle clarification question selection in template modification mode."""
    try:
        question_num = int(option.split("Question ")[1].split(":")[0])
        question_index = question_num - 1
    except (ValueError, IndexError):
        st.session_state.error_message = "Error: Could not parse question number. Please try again."
        st.rerun()
        return
    
    if question_index < 0 or question_index >= len(st.session_state.template_parsed_questions):
        st.session_state.error_message = "Error: Invalid question number. Please try again."
        st.rerun()
        return
    
    st.session_state.current_question_index = question_index
    st.session_state.current_step = "answering_question"
    st.rerun()

def handle_question_answer(answer: str):
    """Handle user's answer to a clarifying question."""
    current_index = st.session_state.current_question_index
    question_data = st.session_state.parsed_questions[current_index]
    
    # Store the answer
    st.session_state.question_answers[current_index] = {
        'question': question_data['question'],
        'answer': answer,
        'keyword': question_data.get('keyword'),
        'example': question_data.get('example')
    }
    
    # Check if there are more questions to answer
    if current_index + 1 < len(st.session_state.parsed_questions):
        # Ask the next question
        st.session_state.current_question_index = current_index + 1
        st.rerun()
    else:
        # All questions answered, create enhanced goal and proceed
        # Create enhanced goal with answers
        enhanced_goal = create_enhanced_goal_with_answers()
        st.session_state.enhanced_goal = enhanced_goal
        
        # Proceed to decomposition with enhanced goal
        proceed_to_decomposition()

def create_enhanced_goal_with_answers():
    """Create an enhanced goal by integrating clarifying question answers."""
    base_goal = st.session_state.detailed_goal or st.session_state.goal
    
    if not st.session_state.question_answers:
        return base_goal
    
    # Create a summary of answers
    answers_summary = "\n\n**Additional Details:**\n"
    for index, answer_data in st.session_state.question_answers.items():
        answers_summary += f"- {answer_data['question']}: {answer_data['answer']}\n"
    
    return base_goal + answers_summary

def proceed_to_generation():
    """Proceed to agent generation."""
    st.session_state.current_step = "final"
    
    st.session_state.final_instructions = st.session_state.current_decomposition
    st.session_state.final_instructions_json = st.session_state.current_decomposition_json
    
    st.rerun()

def generate_agent():
    """Generate the final agent."""
    current_instructions = st.session_state.final_instructions_json or st.session_state.final_instructions
    st.session_state.error_message = None  # Clear previous errors
    
    with st.spinner("Generating your agent..."):
        try:
            agent_json, error = asyncio.run(
                generate_agent_json_from_subtasks(
                    current_instructions,
                    blocks
                )
            )
            
            if error:
                st.session_state.error_message = f"Error generating agent: {error}"
                st.rerun()
                return
            
            # Success - agent generated
            st.session_state.agent_json = agent_json
            st.session_state.working_agent_json = agent_json
            
            # Save agent
            agent_name = agent_json.get("name", "agent")
            filename = re.sub(r'[^a-zA-Z0-9]+', '_', agent_name).strip('_')[:50]
            agent_json_path = OUTPUT_DIR / f"{filename}.json"
            
            try:
                with open(agent_json_path, "w", encoding="utf-8") as f:
                    json.dump(agent_json, f, indent=2, ensure_ascii=False)
            except Exception as e:
                st.warning(f"⚠️ Warning: Could not save agent file: {e}")
            
            # Display results
            st.session_state.current_step = "agent_results"
            st.rerun()
            
        except Exception as e:
            st.session_state.error_message = f"Error during generation: {str(e)}"
            st.rerun()

def generate_updated_agent():
    """Generate the updated agent based on the new instructions."""
    current_instructions = st.session_state.updated_instructions_json or st.session_state.updated_instructions
    st.session_state.error_message = None  # Clear previous errors
    
    with st.spinner("Generating your updated agent..."):
        try:
            agent_json, error = asyncio.run(
                update_agent_json_incrementally(
                    current_instructions,
                    st.session_state.working_agent_json,  # Use the working agent JSON
                    blocks
                )
            )
            
            if error:
                st.session_state.error_message = f"Error generating updated agent: {error}"
                st.rerun()
                return
            
            # Success - updated agent generated
            st.session_state.agent_json = agent_json
            st.session_state.working_agent_json = agent_json  # Update working agent JSON for next improvement iteration
            st.session_state.generation_counter += 1  # Increment generation counter
            
            # Save agent
            agent_name = agent_json.get("name", "agent")
            filename = re.sub(r'[^a-zA-Z0-9]+', '_', agent_name).strip('_')[:50]
            agent_json_path = OUTPUT_DIR / f"{filename}.json"
            
            try:
                with open(agent_json_path, "w", encoding="utf-8") as f:
                    json.dump(agent_json, f, indent=2, ensure_ascii=False)
            except Exception as e:
                st.warning(f"⚠️ Warning: Could not save agent file: {e}")
            
            # Display results
            st.session_state.current_step = "agent_results"
            st.rerun()
            
        except Exception as e:
            st.session_state.error_message = f"Error during updated generation: {str(e)}"
            st.rerun()

def handle_improvement_request(improvement_request: str):
    """Handle agent improvement request."""
    st.session_state.improvement_request = improvement_request
    st.session_state.current_agent_json = st.session_state.agent_json
    # Set the working agent JSON to the most recent one (this will be updated with each improvement)
    st.session_state.working_agent_json = st.session_state.agent_json
    st.session_state.error_message = None  # Clear previous errors
    
    with st.spinner("Processing improvement request..."):
        try:
            # Use the most recent instructions - either updated_instructions from previous chat or original
            current_instructions = st.session_state.updated_instructions_json or st.session_state.final_instructions_json or st.session_state.current_decomposition_json
            
            updated_instructions = asyncio.run(
                update_decomposition_incrementally(
                    improvement_request,
                    current_instructions,
                    block_summaries
                )
            )
            
            if not updated_instructions:
                st.session_state.error_message = "Failed to update instructions"
                st.rerun()
                return
            
            # Check if updated_instructions contains clarifying questions (new JSON format)
            if isinstance(updated_instructions, dict) and updated_instructions.get("type") == "clarifying_questions":
                st.session_state.chat_clarifying_questions = updated_instructions
                st.session_state.chat_parsed_questions = updated_instructions.get("questions", [])
                # Switch step so option clicks are handled by clarification flow
                st.session_state.current_step = "clarification"
                st.rerun()
                
            # Handle unachievable improvement (new JSON format)
            elif isinstance(updated_instructions, dict) and updated_instructions.get("type") == "unachievable_goal":
                st.session_state.last_decomposition = updated_instructions
                st.session_state.current_step = "goal_suggestion"
                st.rerun()
                
            # Handle vague improvement (new JSON format)
            elif isinstance(updated_instructions, dict) and updated_instructions.get("type") == "vague_goal":
                st.session_state.last_decomposition = updated_instructions
                st.session_state.current_step = "goal_suggestion"
                st.rerun()
                
            # Check for old text format
            elif isinstance(updated_instructions, str) and "❓ Clarifying Questions:" in updated_instructions:
                st.session_state.chat_clarifying_questions = updated_instructions
                st.session_state.chat_parsed_questions = parse_clarifying_questions(updated_instructions)
                st.session_state.current_step = "clarification"
                st.rerun()
                
            # Handle updated instructions (new JSON format)
            elif isinstance(updated_instructions, dict) and updated_instructions.get("type") == "instructions":
                instructions_text = "**Updated Step-by-Step Instructions:**\n"
                for step in updated_instructions.get("steps", []):
                    instructions_text += f"{step.get('step_number', '')}. {step.get('description', '')}\n"
                    if step.get('inputs'):
                        instructions_text += "   Inputs:\n"
                        for input_item in step['inputs']:
                            instructions_text += f"   - {input_item.get('name', '')}: {input_item.get('value', '')}\n"
                    if step.get('outputs'):
                        instructions_text += "   Outputs:\n"
                        for output_item in step['outputs']:
                            instructions_text += f"   - {output_item.get('name', '')}: {output_item.get('description', '')}\n"
                    instructions_text += "\n"
                
                # Store both formatted text for UI and raw JSON for agent generation
                st.session_state.updated_instructions = instructions_text
                st.session_state.updated_instructions_json = updated_instructions
                st.session_state.original_instructions = current_instructions
                st.session_state.current_step = "decomposition_review"
                st.rerun()
                
            # Handle old text format instructions
            else:
                st.session_state.updated_instructions = updated_instructions
                st.session_state.original_instructions = current_instructions
                st.session_state.current_step = "decomposition_review"
                st.rerun()
                
        except Exception as e:
            st.session_state.error_message = f"Error processing improvement request: {str(e)}"
            st.rerun()

def handle_improvement_question_answer(answer: str):
    """Handle user's answer to a clarifying question in improvement mode."""
    current_index = st.session_state.current_question_index
    question_data = st.session_state.chat_parsed_questions[current_index]
    
    # Store the answer
    st.session_state.chat_question_answers[current_index] = {
        'question': question_data['question'],
        'answer': answer,
        'keyword': question_data.get('keyword'),
        'example': question_data.get('example')
    }
    
    # Check if there are more questions to answer
    if current_index + 1 < len(st.session_state.chat_parsed_questions):
        # Ask the next question
        st.session_state.current_question_index = current_index + 1
        st.rerun()
    else:
        # All questions answered, create enhanced improvement request and proceed
        # Create enhanced improvement request with answers
        enhanced_request = create_enhanced_improvement_request_with_answers()
        
        # Process the enhanced request
        process_enhanced_improvement_request(enhanced_request)

def handle_template_question_answer(answer: str):
    """Handle user's answer to a clarifying question in template modification mode."""
    current_index = st.session_state.current_question_index
    question_data = st.session_state.template_parsed_questions[current_index]
    
    # Store the answer
    st.session_state.template_question_answers[current_index] = {
        'question': question_data['question'],
        'answer': answer,
        'keyword': question_data.get('keyword'),
        'example': question_data.get('example')
    }
    
    # Check if there are more questions to answer
    if current_index + 1 < len(st.session_state.template_parsed_questions):
        # Ask the next question
        st.session_state.current_question_index = current_index + 1
        st.rerun()
    else:
        # All questions answered, create enhanced modification request and proceed
        # Create enhanced modification request with answers
        enhanced_request = create_enhanced_template_modification_request_with_answers()
        
        # Process the enhanced request
        process_enhanced_template_modification_request(enhanced_request)

def create_enhanced_improvement_request_with_answers():
    """Create an enhanced improvement request by integrating clarifying question answers."""
    base_request = st.session_state.improvement_request
    
    if not st.session_state.chat_question_answers:
        return base_request
    
    # Create a summary of answers
    answers_summary = "\n\n**Additional Details:**\n"
    for index, answer_data in st.session_state.chat_question_answers.items():
        answers_summary += f"- {answer_data['question']}: {answer_data['answer']}\n"
    
    return base_request + answers_summary

def create_enhanced_template_modification_request_with_answers():
    """Create an enhanced template modification request by integrating clarifying question answers."""
    base_request = st.session_state.template_modification_instructions
    
    if not st.session_state.template_question_answers:
        return base_request
    
    # Create a summary of answers
    answers_summary = "\n\n**Additional Details:**\n"
    for index, answer_data in st.session_state.template_question_answers.items():
        answers_summary += f"- {answer_data['question']}: {answer_data['answer']}\n"
    
    return base_request + answers_summary

def process_enhanced_improvement_request(enhanced_request: str):
    """Process the enhanced improvement request with answers."""
    st.session_state.error_message = None  # Clear previous errors
    with st.spinner("Processing enhanced improvement request..."):
        try:
            # Use the most recent instructions - either updated_instructions from previous chat or original
            current_instructions = st.session_state.updated_instructions_json or st.session_state.final_instructions_json or st.session_state.current_decomposition_json
            
            updated_instructions = asyncio.run(
                update_decomposition_incrementally(
                    enhanced_request,
                    current_instructions,
                    block_summaries
                )
            )
            
            if not updated_instructions:
                st.session_state.error_message = "Failed to update instructions"
                st.rerun()
                return
            
            # Handle the updated instructions (should be instructions now, not questions)
            if isinstance(updated_instructions, dict) and updated_instructions.get("type") == "instructions":
                instructions_text = "**Updated Step-by-Step Instructions:**\n"
                for step in updated_instructions.get("steps", []):
                    instructions_text += f"{step.get('step_number', '')}. {step.get('description', '')}\n"
                    if step.get('inputs'):
                        instructions_text += "   Inputs:\n"
                        for input_item in step['inputs']:
                            instructions_text += f"   - {input_item.get('name', '')}: {input_item.get('value', '')}\n"
                    if step.get('outputs'):
                        instructions_text += "   Outputs:\n"
                        for output_item in step['outputs']:
                            instructions_text += f"   - {output_item.get('name', '')}: {output_item.get('description', '')}\n"
                    instructions_text += "\n"
                
                # Store both formatted text for UI and raw JSON for agent generation
                st.session_state.updated_instructions = instructions_text
                st.session_state.updated_instructions_json = updated_instructions
                st.session_state.original_instructions = current_instructions
                st.session_state.current_step = "decomposition_review"
                st.rerun()
                
            else:
                st.session_state.updated_instructions = updated_instructions
                st.session_state.original_instructions = current_instructions
                st.session_state.current_step = "decomposition_review"
                st.rerun()
                
        except Exception as e:
            st.session_state.error_message = f"Error processing enhanced improvement request: {str(e)}"
            st.rerun()

def process_enhanced_template_modification_request(enhanced_request: str):
    """Process the enhanced template modification request with answers."""
    st.session_state.error_message = None  # Clear previous errors
    with st.spinner("Processing enhanced template modification request..."):
        try:
            # Generate instructions based on the template and enhanced modification request
            updated_instructions = asyncio.run(
                generate_template_modification_instructions(
                    st.session_state.template_agent_json,
                    enhanced_request,
                    block_summaries
                )
            )
            
            if not updated_instructions:
                st.session_state.error_message = "Failed to generate modification instructions"
                st.rerun()
                return
            
            # Handle the updated instructions (should be instructions now, not questions)
            if isinstance(updated_instructions, dict) and updated_instructions.get("type") == "instructions":
                instructions_text = "**Modified Step-by-Step Instructions:**\n"
                for step in updated_instructions.get("steps", []):
                    instructions_text += f"{step.get('step_number', '')}. {step.get('description', '')}\n"
                    if step.get('inputs'):
                        instructions_text += "   Inputs:\n"
                        for input_item in step['inputs']:
                            instructions_text += f"   - {input_item.get('name', '')}: {input_item.get('value', '')}\n"
                    if step.get('outputs'):
                        instructions_text += "   Outputs:\n"
                        for output_item in step['outputs']:
                            instructions_text += f"   - {output_item.get('name', '')}: {output_item.get('description', '')}\n"
                    instructions_text += "\n"
                
                # Store both formatted text for UI and raw JSON for agent generation
                st.session_state.updated_instructions = instructions_text
                st.session_state.updated_instructions_json = updated_instructions
                st.session_state.current_step = "template_modification_review"
                st.rerun()
                
            else:
                st.session_state.updated_instructions = updated_instructions
                st.session_state.updated_instructions_json = updated_instructions
                st.session_state.current_step = "template_modification_review"
                st.rerun()
                
        except Exception as e:
            st.session_state.error_message = f"Error processing enhanced template modification request: {str(e)}"
            st.rerun()

def generate_modified_agent_from_template():
    """Generate the modified agent based on template and modification instructions."""
    current_instructions = st.session_state.updated_instructions_json or st.session_state.updated_instructions
    st.session_state.error_message = None  # Clear previous errors
    
    with st.spinner("Generating your modified agent..."):
        try:
            agent_json, error = asyncio.run(
                update_agent_json_incrementally(
                    current_instructions,
                    st.session_state.template_agent_json,  # Use template as base
                    blocks
                )
            )
            
            if error:
                st.session_state.error_message = f"Error generating modified agent: {error}"
                st.rerun()
                return
            
            # Success - modified agent generated
            st.session_state.agent_json = agent_json
            st.session_state.working_agent_json = agent_json
            st.session_state.generation_counter += 1
            
            # Save agent
            agent_name = agent_json.get("name", "agent")
            filename = re.sub(r'[^a-zA-Z0-9]+', '_', agent_name).strip('_')[:50]
            agent_json_path = OUTPUT_DIR / f"{filename}.json"
            
            try:
                with open(agent_json_path, "w", encoding="utf-8") as f:
                    json.dump(agent_json, f, indent=2, ensure_ascii=False)
            except Exception as e:
                st.warning(f"⚠️ Warning: Could not save agent file: {e}")
            
            # Display results
            st.session_state.current_step = "agent_results"
            st.rerun()
            
        except Exception as e:
            st.session_state.error_message = f"Error during generation: {str(e)}"
            st.rerun()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    global block_summaries, blocks
    block_summaries, blocks = load_blocks()
    
    if block_summaries is None:
        st.error("❌ Failed to load blocks. Please check your configuration.")
        st.stop()
    
    # Render the current stage
    render_current_stage()

if __name__ == "__main__":
    main()
