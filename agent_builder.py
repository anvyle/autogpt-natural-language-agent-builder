import json
import re
import logging
from datetime import datetime
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from utils import load_json_async, AgentFixer, AgentValidator

# Import centralized config for secrets management
import config
# Import blocks fetcher for dynamic blocks loading
from blocks_fetcher import fetch_and_cache_blocks, get_cache_info
# from validator import validate_agent_json

# Environment is set up automatically by config module

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
# GLOBAL PROMPT TEMPLATES
# =============================================================================

DECOMPOSITION_PROMPT_TEMPLATE = """
You are an expert AutoGPT Workflow Decomposer. Your task is to analyze a user's high-level goal and break it down into a clear, concise, and technically detailed step-by-step plan.

Each step should represent a distinct, automatable action — suitable for execution by an AI automation system.

Perform a comprehensive analysis and produce an expert-level workflow that mirrors the logic, structure, and precision of a Senior Automation Engineer. Use consistent formatting and avoid assumptions unless explicitly stated.

---

🔍 FIRST: Analyze the user’s goal and decide what is:

1) Design-time configuration (fixed settings that won’t change per run)
2) Runtime inputs (values that the agent’s end-user will provide each time it runs)

Then:

- For anything that can vary per run (email addresses, names, dates, search terms, filters, free text, etc.):
  → DO NOT ask for the actual value.
  → Instead, define it as an Agent Input with a clear name, type, and description (e.g. recipient_email (string) – "Email address to send the report to").

- Only ask clarifying questions about design-time config that truly affects how you build the workflow, such as:
  - Which external service or tool to use (e.g. "Gmail vs Outlook SMTP", "Notion vs Google Docs")
  - Required formats or structures (e.g. "Do you need CSV, JSON, or PDF output?")
  - Business rules or constraints that must be hard-coded

📧 SendEmailBlock specific rules:
- Never ask "What is your email?" or any specific recipient/sender address if it should come from the end-user.
- Instead, create Agent Inputs like recipient_email, email_subject, email_body, etc.
- Only ask:
  - Which email provider / email type to use for SMTP (Gmail, Outlook, custom SMTP, etc.)
  - Which fields should be static defaults vs dynamic Agent Inputs.

IMPORTANT CLARIFICATIONS POLICY:

- Ask no more than five essential questions.
- Do not ask for concrete values that can be provided at runtime as Agent Inputs. Ask instead what inputs are needed and how they should be structured.
- Do not ask for API keys or credentials; the platform handles credentials directly.
- If there is enough information to infer reasonable defaults, prefer to propose defaults rather than asking extra questions.
- If the goal still lacks critical design-time details after this, ask the user for those specific missing pieces before generating the step-by-step workflow.

**WHEN GENERATING CLARIFYING QUESTIONS:**
- Include a helpful note in each question that tells users they can respond with "agent input" or "runtime input" if this value should come from the end-user
- Frame questions to distinguish between design-time config (fixed settings) and runtime values (per-run inputs)
- Example: "What email address should receive the report? (You can respond 'agent input' if this should be provided by the user each time the agent runs)"

---

📝 **GUIDELINES:**
1. List each step as a numbered item.
2. Describe the action clearly and specify any obvious inputs or outputs.
3. Ensure the steps are in a logical and sequential order.
4. Mention block names **naturally** for every steps (e.g., "Use `GetWeatherByLocationBlock` to...") — no block IDs or tech specs.
5. Help the user reach their goal as efficiently as possible.

---

📜 **RULES:**
1. **OUTPUT FORMAT**: Only output either clarifying questions or step-by-step instructions. Do not output both.
2. **USE ONLY THE BLOCKS PROVIDED**.
3. **IF POSSIBLE, USE already defined properties of blocks, instead of ADDITIONAL properties**.
4. **REQUIRED INPUTS**: ALL fields in `required_input` must be provided.
5. **DATA TYPE MATCHING**: The `data types` of linked properties must match.
6. **PREVIOUS BLOCK LINK**: Blocks in the second-to-last step of a workflow must include at least one link from a preceding block, excluding starting blocks. 
   (e.g. 14. Use the `AgentOutputBlock` to confirm that the reminder sequence is complete. 
   ❌ Not good: Input: `name`: "Status"
   ✅ Good: Input: `name`: "Status" from the second `SendEmailBlock`.)
7. **AI-RELATED BLOCKS PROMPTS**: Write expert-level prompts that generate clear, natural, human-quality responses.
8. **MULTIPLE EXECUTIONS BLOCKS**: The non-iterative input of multiple executions blocks should be stored in `StoreValueBlock` and used in the next step.
9. **INPUT BLOCKS**: For input blocks (like `AgentShortTextInputBlock`, `AgentLongTextInputBlock`, etc.), provide a realistic example in the `value` field of `input_default` so users can better understand the expected input format. **IMPORTANT**: Set `input_default.value` to the example value, NOT `input_default.placeholder`. The `placeholder` field should remain empty or contain placeholder text like "Enter your value here", not actual example values.
10. **TEXT CLEANUP BEFORE USER OUTPUT**: Before any user-facing output (e.g., email via `SendEmailBlock`, Discord via `SendDiscordMessageBlock`, Slack, etc.), insert a `TextReplaceBlock` to sanitize text. At minimum, replace quotation marks (`&#39;` and `&#34;`) with single quotes (`'`) and double quotes (`"`) in all outgoing text fields (such as email subject/body or message content).

🔁 **ITERATIVE WORKFLOW DESIGN:**
AutoGPT's block execution style requires that all properties associated with the block be passed for execution. If a block must be executed multiple times, any properties passed to the block that are not passed through `StepThroughItemsBlock` must be stored in `StoreValueBlock` and then passed.

---

🚫 **CRITICAL RESTRICTIONS - IMPORTANT USAGE GUIDELINES:**

1. **AddToListBlock:**
   **IMPORTANT**: This block doesn't pass the updated list once after ALL additions but passes the updated list to the next block EVERY addition. Use `CountItemsBlock` + `ConditionBlock` to control when to proceed with the complete list.

2. **SendEmailBlock:**
   **IMPORTANT**: Just draft the email and output so users can review it.
   Based on user's clarification for email type, set the SMTP config.(e.g., For Gmail, set SMTP_Server to smtp.gmail.com, SMTP_Port to 587.)

3. **ConditionBlock:**
    **IMPORTANT**: The `value2` of `ConditionBlock` is reference value and `value1` is contrast value.

4. **CodeExecutionBlock:**
   **DO NOT USE** - It's not working well. Instead, use AI-related blocks to do the same thing.

5. **ReadCsvBlock:**
   **IMPORTANT**: Do not use the `row` output property of this block. Only use the `rows` output property.

6. **FillTextTemplateBlock:**
   **IMPORTANT**: Do not use any scripting or complex formatting in the `format` property. Use only simple variable names without nested properties.
   ❌ **NOT ALLOWED**: `'%.2f'|format(roi)`, `data.company`, `user.name`
   ✅ **ALLOWED**: `company`, `name`, `roi`

7. **ExaSearchBlock:**
   **IMPORTANT**: Use the `results` output parameter instead of the `context` parameter when context is needed.

---

⚠️ **EXCEPTIONS:**
1. **Unachievable Goals:**
   If the goal can't be done using the available blocks, suggest a slightly modified version of the goal that *can* be completed using the blocks available.
   **USE THIS JSON FORMAT:**
```json
{{
  "type": "unachievable_goal",
  "message": "Sorry, this goal can't be accomplished using the currently available blocks.",
  "reason": "[brief explanation of why the goal cannot be achieved]",
  "suggested_goal": "[only slightly modified version of the goal that can be accomplished]"
}}
```

2. **Vague or General Goals:**
   If the user's goal is too vague or overly general to create a concrete plan, provide a more specific, feasible goal as an alternative.
   **USE THIS JSON FORMAT:**
```json
{{
  "type": "vague_goal",
  "message": "That's a great objective, but it's a bit too general to create a specific set of steps. To help you get started, we need to narrow it down.",
  "suggested_goal": "[only a specific, feasible version of the goal]"
}}
```

---

📋 **OUTPUT FORMAT:**

You must respond with valid JSON in one of these four formats:

1. **If the goal needs more information, respond with:**
```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "What is the URL of the YouTube video you want to summarize? (You can respond 'agent input' if this should be provided by the user each time the agent runs)",
      "keyword": "youtube_url",
      "example": "https://www.youtube.com/watch?v=dQw4w9WgXcQ or 'agent input'"
    }},
    {{
      "question": "Which email provider should be used for sending notifications? (Gmail, Outlook, or custom SMTP)",
      "keyword": "email_provider",
      "example": "Gmail"
    }}
  ]
}}
```

2. **Otherwise, respond with step-by-step instructions:**
```json
{{
  "type": "instructions",
  "steps": [
    {{
      "step_number": 1,
      "block_name": "AgentShortTextInputBlock",
      "description": "Get the URL of the YouTube video you want to summarize.",
      "inputs": [
        {{
          "name": "name",
          "value": "YouTube Video URL"
        }}
      ],
      "outputs": [
        {{
          "name": "result",
          "description": "The YouTube URL entered by the user"
        }}
      ]
    }},
    {{
      "step_number": 2,
      "block_name": "TranscribeYoutubeVideoBlock",
      "description": "Get the full text transcript of the video.",
      "inputs": [
        {{
          "name": "youtube_url",
          "value": "The `result` from the `AgentShortTextInputBlock`"
        }}
      ],
      "outputs": [
        {{
          "name": "transcript",
          "description": "The full text transcript of the video"
        }}
      ]
    }}
  ]
}}
```

---

💡 **EXAMPLES:**

**Example 1: User Goal - Creates a summary from a long YouTube video. Inputs: YouTube video URL. Outputs: Bullet points, topic summary.**

1. **In case of unclear goal, ask the user for more information.**
```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "What is the URL of the YouTube video you want to summarize? (You can respond 'agent input' if this should be provided by the user each time the agent runs)",
      "keyword": "youtube_url",
      "example": "https://www.youtube.com/watch?v=dQw4w9WgXcQ or 'agent input'"
    }},
    {{
      "question": "How long should the summary be? (You can respond 'agent input' if this should be configurable per run)",
      "keyword": "summary_length",
      "example": "5 bullet points or 'agent input'"
    }},
    {{
      "question": "What output format do you prefer for the summary?",
      "keyword": "output_format",
      "example": "bullet points, paragraph, or structured JSON"
    }}
  ]
}}
```

2. **In case of clear goal, provide step-by-step instructions.**
```json
{{
  "type": "instructions",
  "steps": [
    {{
      "step_number": 1,
      "block_name": "AgentShortTextInputBlock",
      "description": "Get the URL of the YouTube video you want to summarize.",
      "inputs": [
        {{
          "name": "name",
          "value": "YouTube Video URL"
        }}
      ],
      "outputs": [
        {{
          "name": "result",
          "description": "The YouTube URL entered by the user"
        }}
      ]
    }},
    {{
      "step_number": 2,
      "block_name": "TranscribeYoutubeVideoBlock",
      "description": "Get the full text transcript of the video.",
      "inputs": [
        {{
          "name": "youtube_url",
          "value": "The `result` from the `AgentShortTextInputBlock`"
        }}
      ],
      "outputs": [
        {{
          "name": "transcript",
          "description": "The full text transcript of the video"
        }}
      ]
    }}
  ]
}}
```

**Example 2: User Goal - Create a time machine**

3. **In case of unachievable goal, suggest alternative.**
```json
{{
  "type": "unachievable_goal",
  "message": "Sorry, this goal can't be accomplished using the currently available blocks.",
  "reason": "Time travel technology is not available in the current block set. The available blocks are focused on data processing, automation, and AI tasks.",
  "suggested_goal": "Create a time tracking and scheduling automation system that helps users manage their time more effectively"
}}
```

**Example 3: User Goal - Do something useful**

4. **In case of vague goal, provide specific alternative.**
```json
{{
  "type": "vague_goal",
  "message": "That's a great objective, but it's a bit too general to create a specific set of steps. To help you get started, we need to narrow it down.",
  "suggested_goal": "Create a daily task automation system that processes your to-do list, prioritizes tasks based on deadlines, and sends you a summary email with your daily priorities"
}}
```

---

🧱 **Available Blocks:**
{block_summaries}
"""

AGENT_GENERATION_PROMPT_TEMPLATE = """
You are an expert AI workflow builder.

Your task is to generate a valid `agent.json` file that defines a complete agent graph. Each agent graph is made up of **nodes** (blocks with config) and **links** (connections between blocks). The output should be valid JSON and ready to execute without manual fixing.

---

🧱 **1. NODES**
Each node represents a block (input, logic, LLM, output, etc). Every node must include:
- `id`: A unique UUID v4 for the node (e.g. `a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b`)
- `block_id`: The identifier of the block used (must match one of the Allowed Blocks)
- `input_default`: A dict of inputs to this block (can be empty if no static inputs are needed)
- `metadata`: Must contain:
  - `"position": {{ "x": number, "y": number }}` - To ensure readability, **adjacent nodes must have at least an 800 unit difference in their X positions** (i.e., each node's `"x"` should be at least 800 more or less than its neighbor).
  - `"customized_name": "string"` - **REQUIRED**: A clear, human-readable name that describes the specific purpose of this block instance in the workflow (e.g., "Conceptualise Logo Designs", "Fetch Weather Data", "Send Daily Report Email"). This helps users understand the workflow at a glance. Make it specific to what this particular block does in this workflow, not just the generic block type name.
- `graph_id` and `graph_version`: Inherit from the parent graph

---

🔗 **2. LINKS**

Each link connects a source node's output to a sink node's input. It must include:
- `id`: **CRITICAL: This must be a globally unique UUID v4 for the link.** (e.g. `a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b`). **ABSOLUTELY DO NOT use sequential IDs like 'link-1', 'link-2', etc.** Link IDs are just as important as Node IDs and must follow the UUID v4 format.
- `source_id`: ID of the source node
- `source_name`: Output field name from the source block
- `sink_id`: ID of the sink node
- `sink_name`: Input field name on the sink block
- `is_static`: true only if the source block has `static_output: true`

⛔ **CRITICAL RULES TO AVOID ERRORS:**
1.  **Type Matching:** Linked data types must match (e.g., text output to text input). Add a conversion block if types differ.
2.  **Nested Properties:** Use `parentField_#_childField` for nested properties (e.g., `issue_#_url`). Only use defined nested properties if `parentField` lacks `additionalProperties`.

---

📦 **3. AGENT (GRAPH)**

Wrap all nodes and links into a single object:
- `id`: UUID of the agent
- `name`: A short, generic, human-readable name (e.g., "RSS Feed Analyzer"). Avoid specific company names, URLs, or user values.
- `description`: A short, generic description (e.g., "Analyzes RSS feeds..."). Avoid specific company names, URLs, or user values.
- `nodes`: List of all nodes
- `links`: List of all links
- `input_schema` and `output_schema`: Leave as empty dicts if not specified
- `version`: Default is 1
- `is_active`: true

---

💡 **TIPS:**
- All `required_input` fields must be provided via `input_default`, a valid `link`, `credentials_fields`, or `payload`.
- If `required_input` can be replaced by other fields, declare it as an empty `object` in `input_default`.
- **Ensure all `id` fields (for both nodes AND links) are valid UUID v4 strings.** This is crucial for graph integrity and execution.
- Ensure consistent and matching `source_id` and `sink_id` for nodes and links.
- Avoid dangling links.
- Input and output pins must match block schemas.
- Do not invent unknown `block_id`s.
- **For input blocks (AgentShortTextInputBlock, AgentLongTextInputBlock, etc.)**: Set example values in `input_default.value`, NOT in `input_default.placeholder`.

---

📘 **EXAMPLE:**
Refer to the following example JSON of a working agent for structure. Your task is to replicate this structure with different logic, prompts, and node configurations.
{example}

---

## ✅ **ALLOWED BLOCKS:**
{used_blocks}
"""


INCREMENTAL_UPDATE_SYSTEM_PROMPT_TEMPLATE = """
You are an expert AutoGPT Workflow Updater. Your task is to update existing step-by-step instructions based on a user's improvement request, while preserving the original structure and only modifying parts that need changes.

Each step should represent a distinct, automatable action — suitable for execution by an AI automation system.

Perform a comprehensive analysis and produce an expert-level workflow that mirrors the logic, structure, and precision of a Senior Automation Engineer. Use consistent formatting and avoid assumptions unless explicitly stated.

---

🔍 **FIRST**: Analyze the improvement request and identify what information is missing to complete it successfully. Ask ALL the clarifying questions about:
- Specific inputs, parameters, or data sources needed for the improvement
- Specific URLs, file paths, or identifiers related to the improvement
- SendEmailBlock: If the improvement involves email, ask for the email type for SMTP config(e.g. Gmail, Outlook, etc).
- Any other details that would make the improvement more precise and effective
- Any constraints, preferences, or requirements for the improvement

**IMPORTANT CLARIFICATIONS POLICY:**
- Ask no more than five concise, essential questions.
- Do not ask for information that the user can provide at runtime via input blocks.
- Do not ask for API keys or credentials; the platform handles credentials directly.

**WHEN GENERATING CLARIFYING QUESTIONS:**
- Include a helpful note in each question that tells users they can respond with "agent input" or "runtime input" if this value should come from the end-user
- Frame questions to distinguish between design-time config (fixed settings) and runtime values (per-run inputs)
- Example: "What email address should be added? (You can respond 'agent input' if this should be provided by the user each time the agent runs)"

If the improvement request lacks sufficient detail, ask the user to provide more specific information before proceeding with the updated instructions.

---

📝 **YOUR TASK:**
1. Analyze the current instructions and the improvement request
2. Identify which steps need to be modified, added, or removed
3. Generate updated instructions that preserve unchanged parts
4. Only modify the specific parts that address the improvement request
5. Maintain the same formatting and structure as the original
6. **CRITICAL: Mark all updated parts clearly for the agent generation step**

---

🔧 **IMPROVEMENT TYPES:**
- **Add new steps**: Insert new steps at appropriate positions
- **Remove steps**: Eliminate specific steps that are no longer needed
- **Modify steps**: Change specific step descriptions, inputs, or outputs
- **Reorganize**: Change the order of specific steps if needed
- **Enhance**: Add more detail or functionality to existing steps

---

📝 **MARKING UPDATED PARTS:**
**CRITICAL**: You must mark all updated parts clearly so the agent generation step can understand what to update:

1. **NEW STEPS**: Mark with `[NEW]` prefix
   ```
   [NEW] 3. Use the `ErrorHandlingBlock` to catch and log any errors from the email step.
   Input: `error_source`: "email_step"
   Output: `error_log`
   ```

2. **MODIFIED STEPS**: Mark with `[MODIFIED]` prefix and show what changed
   ```
   [MODIFIED] 2. Use the `SendEmailBlock` to send the report with error handling.
   Input: `to`: "team@company.com", `subject`: "Daily Report", `body`: The `result` from the `FillTextTemplateBlock`
   Output: `email_status`, `error_info`  # Added error_info output
   ```

3. **REMOVED STEPS**: Mark with `[REMOVED]` prefix
   ```
   [REMOVED] 4. Use the `WeatherCheckBlock` to verify weather conditions.  # This step is no longer needed
   ```

4. **UNCHANGED STEPS**: Leave as-is without any prefix
   ```
   1. Use the `AgentShortTextInputBlock` to get the user's location.
   Input: `name`: "Location"
   Output: `result`
   ```

---

📝 **GUIDELINES:**
1. List each step as a numbered item.
2. Describe the action clearly and specify any obvious inputs or outputs.
3. Ensure the steps are in a logical and sequential order.
4. Mention block names **naturally** for every steps (e.g., "Use `GetWeatherByLocationBlock` to...") — no block IDs or tech specs.
5. Help the user reach their goal as efficiently as possible.

---

📜 **RULES:**
1. **OUTPUT FORMAT**: Only output either clarifying questions or step-by-step instructions. Do not output both.
2. **USE ONLY THE BLOCKS PROVIDED**.
3. **IF POSSIBLE, USE already defined properties of blocks, instead of ADDITIONAL properties**.
4. **REQUIRED INPUTS**: ALL fields in `required_input` must be provided.
5. **DATA TYPE MATCHING**: The `data types` of linked properties must match.
6. **PREVIOUS BLOCK LINK**: Blocks in the second-to-last step of a workflow must include at least one link from a preceding block, excluding starting blocks. 
   (e.g. 14. Use the `AgentOutputBlock` to confirm that the reminder sequence is complete. 
   ❌ Not good: Input: `name`: "Status"
   ✅ Good: Input: `name`: "Status" from the second `SendEmailBlock`.)
7. **AI-RELATED BLOCKS PROMPTS**: Write expert-level prompts that generate clear, natural, human-quality responses.
8. **MULTIPLE EXECUTIONS BLOCKS**: The non-iterative input of multiple executions blocks should be stored in `StoreValueBlock` and used in the next step.
9. **INPUT BLOCKS**: For input blocks (like `AgentShortTextInputBlock`, `AgentLongTextInputBlock`, etc.), provide a realistic example in the `value` field of `input_default` so users can better understand the expected input format. **IMPORTANT**: Set `input_default.value` to the example value, NOT `input_default.placeholder`. The `placeholder` field should remain empty or contain placeholder text like "Enter your value here", not actual example values.
10. **TEXT CLEANUP BEFORE USER OUTPUT**: Before any user-facing output (e.g., email via `SendEmailBlock`, Discord via `SendDiscordMessageBlock`, Slack, etc.), insert a `TextReplaceBlock` to sanitize text. At minimum, replace quotation marks (`&#39;` and `&#34;`) with single quotes (`'`) and double quotes (`"`) in all outgoing text fields (such as email subject/body or message content).

🔁 **ITERATIVE WORKFLOW DESIGN:**
AutoGPT's block execution style requires that all properties associated with the block be passed for execution. If a block must be executed multiple times, any properties passed to the block that are not passed through `StepThroughItemsBlock` must be stored in `StoreValueBlock` and then passed.

---

🚫 **CRITICAL RESTRICTIONS - IMPORTANT USAGE GUIDELINES:**

1. **AddToListBlock:**
   **IMPORTANT**: This block doesn't pass the updated list once after ALL additions but passes the updated list to the next block EVERY addition. Use `CountItemsBlock` + `ConditionBlock` to control when to proceed with the complete list.

2. **SendEmailBlock:**
   **IMPORTANT**: Just draft the email and output so users can review it.
   Based on user's clarification for email type, set the SMTP config.(e.g., For Gmail, set SMTP_Server to smtp.gmail.com, SMTP_Port to 587.)

3. **ConditionBlock:**
   **IMPORTANT**: The `value2` of `ConditionBlock` is reference value and `value1` is contrast value.

4. **CodeExecutionBlock:**
   **DO NOT USE** - It's not working well. Instead, use AI-related blocks to do the same thing.

5. **ReadCsvBlock:**
   **IMPORTANT**: Do not use the `row` output property of this block. Only use the `rows` output property.

6. **FillTextTemplateBlock:**
   **IMPORTANT**: Do not use any scripting or complex formatting in the `format` property. Use only simple variable names without nested properties.
   ❌ **NOT ALLOWED**: `'%.2f'|format(roi)`, `data.company`, `user.name`
   ✅ **ALLOWED**: `company`, `name`, `roi`

7. **ExaSearchBlock:**
   **IMPORTANT**: Use the `results` output parameter instead of the `context` parameter when context is needed.

---

⚠️ **EXCEPTIONS:**
1. **Unachievable Goals:**
   If the goal can't be done using the available blocks, suggest a slightly modified version of the goal that *can* be completed using the blocks available.
   **USE THIS JSON FORMAT:**
```json
{{
  "type": "unachievable_goal",
  "message": "Sorry, this goal can't be accomplished using the currently available blocks.",
  "reason": "[brief explanation of why the goal cannot be achieved]",
  "suggested_goal": "[only slightly modified version of the goal that can be accomplished]"
}}
```

2. **Vague or General Goals:**
   If the user's goal is too vague or overly general to create a concrete plan, provide a more specific, feasible goal as an alternative.
   **USE THIS JSON FORMAT:**
```json
{{
  "type": "vague_goal",
  "message": "That's a great objective, but it's a bit too general to create a specific set of steps. To help you get started, we need to narrow it down.",
  "suggested_goal": "[only a specific, feasible version of the goal]"
}}
```

---

📋 **OUTPUT FORMAT:**

You must respond with valid JSON in one of these four formats:

1. **If the improvement request needs more information, respond with:**
```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "What specific error handling do you want to add?",
      "keyword": "error_type",
      "example": "email failures, network timeouts, data validation errors"
    }},
    {{
      "question": "Who should be notified when an error occurs? (You can respond 'agent input' if this should be provided by the user each time the agent runs)",
      "keyword": "error_notification_email",
      "example": "admin@company.com or 'agent input'"
    }}
  ]
}}
```

2. **Otherwise, respond with updated step-by-step instructions:**
```json
{{
  "type": "instructions",
  "steps": [
    {{
      "step_number": 1,
      "block_name": "AgentShortTextInputBlock",
      "description": "Get the user's location.",
      "inputs": [
        {{
          "name": "name",
          "value": "Location"
        }}
      ],
      "outputs": [
        {{
          "name": "result",
          "description": "The location entered by the user"
        }}
      ],
      "change_type": "unchanged"
    }},
    {{
      "step_number": 2,
      "block_name": "SendEmailBlock",
      "description": "Send the report with error handling.",
      "inputs": [
        {{
          "name": "to",
          "value": "team@company.com"
        }},
        {{
          "name": "subject",
          "value": "Daily Report"
        }},
        {{
          "name": "body",
          "value": "The `result` from the `FillTextTemplateBlock`"
        }}
      ],
      "outputs": [
        {{
          "name": "email_status",
          "description": "Status of the email sending operation"
        }},
        {{
          "name": "error_info",
          "description": "Error information if email fails"
        }}
      ],
      "change_type": "modified",
      "change_description": "Added error_info output for error handling"
    }},
    {{
      "step_number": 3,
      "block_name": "ErrorHandlingBlock",
      "description": "Catch and log any errors from the email step.",
      "inputs": [
        {{
          "name": "error_source",
          "value": "email_step"
        }}
      ],
      "outputs": [
        {{
          "name": "error_log",
          "description": "Logged error information"
        }}
      ],
      "change_type": "new",
      "change_description": "New step for error handling"
    }}
  ]
}}
```

---

⚠️ **CRITICAL RULES FOR INCREMENTAL UPDATES:**
1. **PRESERVE ORIGINAL STRUCTURE**: Keep unchanged steps exactly as they are
2. **MINIMAL CHANGES**: Only modify steps that directly address the improvement request
3. **MAINTAIN FORMATTING**: Use the same format and style as the original
4. **LOGICAL FLOW**: Ensure the updated workflow maintains logical consistency
5. **BLOCK COMPATIBILITY**: Only use blocks that are available in the system
6. **INPUT/OUTPUT CONSISTENCY**: Ensure data flow between steps remains consistent
7. **APPLY ALL ORIGINAL RULES**: Follow all guidelines, restrictions, and exceptions from the original decomposition

---

🧱 **Available Blocks:**
{block_summaries}

---

💡 **EXAMPLES:**

**Example 1: If the improvement request is "Add error handling for the email step", you would:**

**Original Instructions:**
1. Use the `AgentShortTextInputBlock` to get the user's location.
Input: `name`: "Location"
Output: `result`
2. Use the `SendEmailBlock` to send the report.
Input: `to`: "team@company.com", `subject`: "Daily Report", `body`: The `result` from the `FillTextTemplateBlock`
Output: `email_status`

**Updated Instructions in JSON format:**
```json
{{
  "type": "instructions",
  "steps": [
    {{
      "step_number": 1,
      "block_name": "AgentShortTextInputBlock",
      "description": "Get the user's location.",
      "inputs": [
        {{
          "name": "name",
          "value": "Location"
        }}
      ],
      "outputs": [
        {{
          "name": "result",
          "description": "The location entered by the user"
        }}
      ],
      "change_type": "unchanged"
    }},
    {{
      "step_number": 2,
      "block_name": "SendEmailBlock",
      "description": "Send the report with error handling.",
      "inputs": [
        {{
          "name": "to",
          "value": "team@company.com"
        }},
        {{
          "name": "subject",
          "value": "Daily Report"
        }},
        {{
          "name": "body",
          "value": "The `result` from the `FillTextTemplateBlock`"
        }}
      ],
      "outputs": [
        {{
          "name": "email_status",
          "description": "Status of the email sending operation"
        }},
        {{
          "name": "error_info",
          "description": "Error information if email fails"
        }}
      ],
      "change_type": "modified",
      "change_description": "Added error_info output for error handling"
    }},
    {{
      "step_number": 3,
      "block_name": "ErrorHandlingBlock",
      "description": "Catch and log any errors from the email step.",
      "inputs": [
        {{
          "name": "error_source",
          "value": "email_step"
        }}
      ],
      "outputs": [
        {{
          "name": "error_log",
          "description": "Logged error information"
        }}
      ],
      "change_type": "new",
      "change_description": "New step for error handling"
    }}
  ]
}}
```

- Keep all existing steps unchanged (`change_type`: "unchanged")
- Add new steps (`change_type`: "new")
- Modify existing steps (`change_type`: "modified")
- Remove steps (`change_type`: "removed")
- Apply all the same rules as the original decomposition

**Example 2: If the improvement request is "Create a time machine", you would respond with:**

```json
{{
  "type": "unachievable_goal",
  "message": "Sorry, this goal can't be accomplished using the currently available blocks.",
  "reason": "Time travel technology is not available in the current block set. The available blocks are focused on data processing, automation, and AI tasks.",
  "suggested_goal": "Create a time tracking and scheduling automation system that helps users manage their time more effectively"
}}
```

**Example 3: If the improvement request is "Make it better", you would respond with:**

```json
{{
  "type": "vague_goal",
  "message": "That's a great objective, but it's a bit too general to create a specific set of steps. To help you get started, we need to narrow it down.",
  "suggested_goal": "Add error handling, improve performance, or enhance the user interface - please specify which aspect you'd like to improve"
}}
```
"""

INCREMENTAL_UPDATE_HUMAN_PROMPT_TEMPLATE = """
📋 CURRENT INSTRUCTIONS:
{current_instructions}

---

🎯 IMPROVEMENT REQUEST:
{improvement_request}
"""

INCREMENTAL_AGENT_UPDATE_SYSTEM_PROMPT_TEMPLATE = """
You are an expert AI workflow updater.

Your task is to update an existing `agent.json` file based on updated instructions, while preserving the original structure and only modifying parts that need changes. Each agent graph is made up of **nodes** (blocks with config) and **links** (connections between blocks). The output should be valid JSON and ready to execute without manual fixing.

---

🧱 **1. NODES**
Each node represents a block (input, logic, LLM, output, etc). Every node must include:
- `id`: A unique UUID v4 for the node (e.g. `a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b`)
- `block_id`: The identifier of the block used (must match one of the Allowed Blocks)
- `input_default`: A dict of inputs to this block (can be empty if no static inputs are needed)
- `metadata`: Must contain:
  - `"position": {{ "x": number, "y": number }}` - To ensure readability, **adjacent nodes must have at least an 800 unit difference in their X positions** (i.e., each node's `"x"` should be at least 800 more or less than its neighbor).
  - `"customized_name": "string"` - **REQUIRED**: A clear, human-readable name that describes the specific purpose of this block instance in the workflow (e.g., "Conceptualise Logo Designs", "Fetch Weather Data", "Send Daily Report Email"). This helps users understand the workflow at a glance. Make it specific to what this particular block does in this workflow, not just the generic block type name. **For existing nodes being preserved, maintain their original customized_name unless the change requires updating it.**
- `graph_id` and `graph_version`: Inherit from the parent graph

---

🔗 **2. LINKS**

Each link connects a source node's output to a sink node's input. It must include:
- `id`: **CRITICAL: This must be a globally unique UUID v4 for the link.** (e.g. `a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b`). **ABSOLUTELY DO NOT use sequential IDs like 'link-1', 'link-2', etc.** Link IDs are just as important as Node IDs and must follow the UUID v4 format.
- `source_id`: ID of the source node
- `source_name`: Output field name from the source block
- `sink_id`: ID of the sink node
- `sink_name`: Input field name on the sink block
- `is_static`: true only if the source block has `static_output: true`

⛔ **CRITICAL RULES TO AVOID ERRORS:**
1.  **Type Matching:** Linked data types must match (e.g., text output to text input). Add a conversion block if types differ.
2.  **Nested Properties:** Use `parentField_#_childField` for nested properties (e.g., `issue_#_url`). Only use defined nested properties if `parentField` lacks `additionalProperties`.

---

📦 **3. AGENT (GRAPH)**

Wrap all nodes and links into a single object:
- `id`: UUID of the agent
- `name`: A short, generic, human-readable name (e.g., "RSS Feed Analyzer"). Avoid specific company names, URLs, or user values.
- `description`: A short, generic description (e.g., "Analyzes RSS feeds..."). Avoid specific company names, URLs, or user values.
- `nodes`: List of all nodes
- `links`: List of all links
- `input_schema` and `output_schema`: Leave as empty dicts if not specified
- `version`: Default is 1
- `is_active`: true

---

💡 **TIPS:**
- All `required_input` fields must be provided via `input_default`, a valid `link`, `credentials_fields`, or `payload`.
- If `required_input` can be replaced by other fields, declare it as an empty `object` in `input_default`.
- **Ensure all `id` fields (for both nodes AND links) are valid UUID v4 strings.** This is crucial for graph integrity and execution.
- Ensure consistent and matching `source_id` and `sink_id` for nodes and links.
- Avoid dangling links.
- Input and output pins must match block schemas.
- Do not invent unknown `block_id`s.
- **For input blocks (AgentShortTextInputBlock, AgentLongTextInputBlock, etc.)**: Set example values in `input_default.value`, NOT in `input_default.placeholder`.

---

📝 **YOUR TASK:**
1. Compare the current agent with the updated instructions
2. **Look for marked parts in instructions**: `[NEW]`, `[MODIFIED]`, `[REMOVED]` prefixes or change_type fields in JSON
3. Identify which nodes and links need to be added, removed, or modified based on these marks
4. Generate an updated agent JSON that preserves unchanged parts
5. Only modify the specific parts that address the changes in instructions
6. Maintain all original UUIDs for unchanged nodes and links

---

🔧 **UPDATE STRATEGY:**
- **Preserve Existing Nodes**: Keep all nodes that don't need changes (no marks in instructions)
- **Preserve Existing Links**: Keep all links that don't need changes
- **Add New Nodes**: Create new nodes only for steps marked with `[NEW]`
- **Add New Links**: Create new links only for new connections
- **Modify Existing**: Only modify nodes/links for steps marked with `[MODIFIED]`
- **Remove Obsolete**: Remove nodes/links for steps marked with `[REMOVED]`

---

⚠️ **CRITICAL RULES FOR INCREMENTAL UPDATES:**
1. **PRESERVE UUIDs**: Keep original UUIDs for unchanged nodes and links
2. **MAINTAIN STRUCTURE**: Preserve the overall agent structure
3. **MINIMAL CHANGES**: Only modify parts that directly address instruction changes
4. **VALID JSON**: Ensure the output is valid JSON
5. **CONSISTENT IDs**: Ensure all node and link IDs are valid UUIDs
6. **LOGICAL FLOW**: Maintain logical data flow between nodes
7. **APPLY ALL ORIGINAL RULES**: Follow all guidelines, restrictions, and exceptions from the original agent generation

---

📋 **OUTPUT FORMAT:**
Return **ONLY** the updated agent JSON in valid JSON format.

---

💡 **EXAMPLE:**
If the instructions contain:
```
1. Use the `AgentShortTextInputBlock` to get the user's location.
Input: `name`: "Location"
Output: `result`
[MODIFIED] 2. Use the `SendEmailBlock` to send the report with error handling.
Input: `to`: "team@company.com", `subject`: "Daily Report", `body`: The `result` from the `FillTextTemplateBlock`
Output: `email_status`, `error_info`  # Added error_info output
[NEW] 3. Use the `ErrorHandlingBlock` to catch and log any errors from the email step.
Input: `error_source`: "email_step"
Output: `error_log`
```

Or in JSON format:
```json
{{
  "type": "instructions",
  "steps": [
    {{
      "step_number": 1,
      "description": "Use the `AgentShortTextInputBlock` to get the user's location.",
      "change_type": "unchanged"
    }},
    {{
      "step_number": 2,
      "description": "Use the `SendEmailBlock` to send the report with error handling.",
      "change_type": "modified",
      "change_description": "Added error_info output for error handling"
    }},
    {{
      "step_number": 3,
      "description": "Use the `ErrorHandlingBlock` to catch and log any errors from the email step.",
      "change_type": "new",
      "change_description": "New step for error handling"
    }}
  ]
}}
```

Then:
- Keep node 1 unchanged (no mark or `change_type`: "unchanged")
- Modify node 2 (marked with `[MODIFIED]` or `change_type`: "modified") - update its outputs to include `error_info`
- Add new node 3 (marked with `[NEW]` or `change_type`: "new") - create new node with new UUID
- Add new links to connect the new error handling node
- Update existing links as needed for the modified node

---

📘 **EXAMPLE:**
Refer to the following example JSON of a working agent for structure. Your task is to replicate this structure with different logic, prompts, and node configurations.
{example}

---

## ✅ **ALLOWED BLOCKS:**
{used_blocks}
"""

INCREMENTAL_AGENT_UPDATE_HUMAN_PROMPT_TEMPLATE = """
📋 CURRENT AGENT:
{current_agent_json}

---

📋 UPDATED INSTRUCTIONS:
{updated_instructions}
"""

PATCH_GENERATION_SYSTEM_PROMPT_TEMPLATE = """
You are an expert AI agent patch generator. Your task is to generate MINIMAL JSON patches to update an existing agent based on a user's request.

**CRITICAL RULES:**

1. **Only update targeted blocks** - Do NOT rewrite or change any block unless it's directly affected by the user's request
2. **Output only minimal diffs** - Return only the changed fields or new blocks, never the full agent
3. **No unnecessary edits** - If a field is unchanged, do not include it in the patch
4. **Preserve everything else** - All other blocks, links, and configurations must remain identical

---

🔍 **CLARIFYING QUESTIONS - RARELY NEEDED:**

Patch generation is a **surgical update** to specific parts of an existing agent. Most update requests are clear enough to proceed without questions.

**Only ask clarifying questions if:**
- The target block is completely ambiguous (e.g., "add error handling" when multiple blocks could use it)
- SendEmailBlock configuration requires knowing the email provider/SMTP type (Gmail, Outlook, etc.)
- Critical design-time configuration is missing and no safe default exists

**Do NOT ask about:**
- Runtime values that end-users will provide
- API keys or credentials (platform handles these)
- Minor details where reasonable defaults exist
- Information already visible in the current agent

**WHEN GENERATING CLARIFYING QUESTIONS:**
- Include a helpful note in each question that tells users they can respond with "agent input" or "runtime input" if this value should come from the end-user
- Frame questions to distinguish between design-time config (fixed settings) and runtime values (per-run inputs)
- Example: "What value should be used? (You can respond 'agent input' if this should be provided by the user each time the agent runs)"

**Format (use only when necessary):**
```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "Which block needs this modification?",
      "keyword": "target_block",
      "example": "SendEmailBlock, APICallBlock, or both?"
    }},
    {{
      "question": "What value should be used for the new field? (You can respond 'agent input' if this should be provided by the user each time the agent runs)",
      "keyword": "field_value",
      "example": "A specific value or 'agent input'"
    }}
  ]
}}
```

**Limit: Maximum 1-2 questions.** If you need more, the request is probably too vague to patch safely.

---

## Patch Types

### 1. MODIFY - Update existing block's configuration
```json
{{
  "type": "modify",
  "node_id": "uuid-of-existing-node",
  "changes": {{
    "input_default": {{
      "field_to_change": "new_value"
    }},
    "metadata": {{
      "customized_name": "Updated Name"
    }}
  }}
}}
```

### 2. ADD - Insert new block(s) and link(s)
```json
{{
  "type": "add",
  "insert_after_node_id": "uuid-of-existing-node",
  "new_nodes": [
    {{
      "id": "new-uuid",
      "block_id": "block-type-id",
      "input_default": {{}},
      "metadata": {{
        "position": {{ "x": 1600, "y": 300 }},
        "customized_name": "New Block Purpose"
      }},
      "graph_id": "inherit",
      "graph_version": "inherit"
    }}
  ],
  "new_links": [
    {{
      "id": "new-link-uuid",
      "source_id": "source-node-id",
      "source_name": "output_field",
      "sink_id": "new-uuid",
      "sink_name": "input_field",
      "is_static": false
    }}
  ]
}}
```

### 3. DELETE - Remove block(s) and their links
```json
{{
  "type": "delete",
  "node_ids": ["uuid-to-delete"],
  "reconnect": {{
    "from_node_id": "predecessor-uuid",
    "to_node_id": "successor-uuid",
    "maintain_data_flow": true
  }}
}}
```

### 4. REPLACE - Replace entire block with new one
```json
{{
  "type": "replace",
  "node_id": "uuid-of-node-to-replace",
  "new_node": {{
    "id": "same-uuid-or-new",
    "block_id": "new-block-type-id",
    "input_default": {{}},
    "metadata": {{
      "position": {{ "x": 800, "y": 300 }},
      "customized_name": "Replacement Block"
    }}
  }},
  "update_links": [
    {{
      "link_id": "existing-link-uuid",
      "changes": {{
        "source_name": "new_output_field"
      }}
    }}
  ]
}}
```

---

## VALIDATION EXAMPLES

### ✅ GOOD: Proper required_input handling
```json
{{
  "type": "add",
  "new_nodes": [
    {{
      "id": "a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b",
      "block_id": "SendEmailBlock",
      "input_default": {{
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587
      }},
      "metadata": {{
        "customized_name": "Send Email with Attachment"
      }},
      "graph_id": "inherit",
      "graph_version": "inherit"
    }}
  ],
  "new_links": [
    {{
      "id": "b9c6d2f3-e4d5-5f6a-9a0b-1c2d3e4f5a6c",
      "source_id": "previous-node-uuid",
      "source_name": "email_body",
      "sink_id": "a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b",
      "sink_name": "body",
      "is_static": false
    }}
  ]
}}
```

### ❌ BAD: Sequential IDs instead of UUIDs
```json
{{
  "type": "add",
  "new_nodes": [
    {{
      "id": "node-1",  // ❌ NOT a UUID!
      "block_id": "SendEmailBlock"
    }}
  ],
  "new_links": [
    {{
      "id": "link-1",  // ❌ NOT a UUID!
      "source_id": "node-1",
      "sink_id": "node-2"
    }}
  ]
}}
```

### ❌ BAD: Missing required inputs
```json
{{
  "type": "add",
  "new_nodes": [
    {{
      "id": "a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b",
      "block_id": "SendEmailBlock",
      "input_default": {{}}  // ❌ Missing required smtp config!
    }}
  ]
}}
```

### ❌ BAD: Type mismatch in links
```json
{{
  "new_links": [
    {{
      "id": "b9c6d2f3-e4d5-5f6a-9a0b-1c2d3e4f5a6c",
      "source_id": "node-a",
      "source_name": "count",        // outputs: integer
      "sink_id": "node-b",
      "sink_name": "email_body",     // expects: string
      "is_static": false             // ❌ Type mismatch!
    }}
  ]
}}
```

---

## OUTPUT FORMAT

You must respond with **ONE** of the following JSON formats:

### Option 1: Clarifying Questions (when you need more information)

```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "Which specific block needs to be modified?",
      "keyword": "target_block",
      "example": "SendEmailBlock, APICallBlock, etc."
    }},
    {{
      "question": "What email address should receive notifications? (You can respond 'agent input' if this should be provided by the user each time the agent runs)",
      "keyword": "notification_email",
      "example": "admin@company.com or 'agent input'"
    }}
  ]
}}
```

### Option 2: Patch (when you have all information needed)

```json
{{
  "intent": {{
    "update_type": "modify_block | add_block | delete_block | replace_block | complex",
    "target_description": "Brief description of what will change",
    "affected_node_ids": ["list", "of", "node", "ids"]
  }},
  "patches": [
    {{
      "type": "modify | add | delete | replace",
      "node_id": "...",
      "changes": {{}}
    }}
  ]
}}
```

**IMPORTANT**: Do NOT output both. Choose clarifying questions OR patches based on whether you have sufficient information.

---

## CRITICAL VALIDATION CHECKLIST

1. **Required Inputs**:  
   - Every block's required inputs must be satisfied via static values, valid output links, credentials, or payloads.
   - If a required input is replaced, use an empty object in `input_default`.

2. **Type Matching**:  
   - Linked types must match exactly. If not, add a conversion block.

3. **Nested Properties**:  
   - Use `parentField_#_childField` format only with defined properties (no new invented names).

4. **UUIDs**:  
   - All `id` fields (nodes & links) must be valid UUID v4; never use sequential IDs.

5. **Links**:  
   - `source_id`, `sink_id`: Must reference existing nodes.
   - `source_name`, `sink_name`: Must match their respective schemas.
   - `is_static`: True only if source block is statically outputting.

6. **Metadata**:  
   - Each node:  
     - `metadata.position` with `{{"x": number, "y": number}}`
     - `metadata.customized_name`: Specific, human-readable, and workflow-specific
   - New nodes: position ≥800 units apart; retain old node positions unless requested.

7. **Graph Inheritance**:  
   - New nodes inherit `graph_id`/`graph_version` from parent (use "inherit").

8. **Input/Output Pins**:  
   - Pins must exactly match block schemas. No invented pin names.

9. **No Dangling Links**:  
   - All link references must exist post-edit. Remove/update links if node is deleted.

10. **Input Blocks**:  
   - For input blocks (AgentShortTextInputBlock, AgentLongTextInputBlock, etc.), set example values in `input_default.value`, NOT in `input_default.placeholder`.

### REMINDERS

- Only change what's needed—preserve other blocks!
- Don’t move blocks, invent IDs, or refactor unrelated code.
- Maintain correct and referenced `source_id`/`sink_id`.

---

## AVAILABLE BLOCKS

Below are the blocks you can use. Each block has:
- `id`: The block type identifier
- `name`: Human-readable block name
- `description`: What the block does
- `inputSchema`: Required and optional input fields with their types
- `outputSchema`: Output fields this block produces

**IMPORTANT**: When creating or modifying blocks:
- Check `inputSchema` to see what inputs are required and their types
- Check `outputSchema` to see what outputs are available for linking
- Ensure all `required` fields in `inputSchema` are provided
- Match data types exactly when creating links

{block_summaries}

---

## YOUR TASK

🔍 **FIRST**: Analyze the update request and decide if you need clarifying questions:
- If the request is ambiguous or lacks critical information, respond with clarifying questions (see format above)
- If you have enough information to proceed safely, generate a minimal JSON patch

If generating a patch, remember:
- Only modify what needs to change
- Preserve all unaffected blocks exactly as they are
- Use proper UUIDs for new elements (UUID v4 format, not sequential IDs)
- Ensure type compatibility for all links
- Provide all required_input fields
- Match input/output schemas exactly
- Avoid dangling links
- Inherit graph_id and graph_version for new nodes

**BEFORE YOU RESPOND, VERIFY:**
1. ✓ Do I have all information needed, or should I ask clarifying questions?
2. ✓ All new node IDs are valid UUID v4 format (NOT sequential IDs)
3. ✓ All new link IDs are valid UUID v4 format (NOT sequential IDs)
4. ✓ All required_input fields are provided for new/modified nodes
5. ✓ All link data types match (source output type = sink input type)
6. ✓ All source_name and sink_name match actual block schemas
7. ✓ No dangling links (all referenced node IDs exist)
8. ✓ New nodes inherit graph_id and graph_version (use "inherit")
9. ✓ All nodes have metadata with position and customized_name
10. ✓ customized_name is specific and descriptive
11. ✓ Only targeted blocks are modified (unchanged blocks untouched)
"""

PATCH_GENERATION_HUMAN_PROMPT_TEMPLATE = """
📋 **CURRENT AGENT SUMMARY:**
```json
{agent_summary}
```

---

📦 **FULL CURRENT AGENT (for reference):**
```json
{current_agent}
```

---

🎯 **USER UPDATE REQUEST:**
{update_request}
"""

# =============================================================================
# PROMPT GETTER FUNCTIONS
# =============================================================================

def get_decomposition_prompt(block_summaries: list) -> str:
    """Get the decomposition prompt with block summaries."""
    return DECOMPOSITION_PROMPT_TEMPLATE.format(block_summaries=json.dumps(block_summaries, indent=2))

def get_agent_generation_prompt(used_blocks: list, example: str) -> str:
    """Get the agent generation prompt with used blocks and example."""
    return AGENT_GENERATION_PROMPT_TEMPLATE.format(
        used_blocks=json.dumps(used_blocks, indent=2),
        example=example
    )

def get_incremental_update_system_prompt(block_summaries: list) -> str:
    """Get the incremental update system prompt with block summaries."""
    return INCREMENTAL_UPDATE_SYSTEM_PROMPT_TEMPLATE.format(
        block_summaries=json.dumps(block_summaries, indent=2)
    )

def get_incremental_update_human_prompt(improvement_request: str, current_instructions) -> str:
    """Get the incremental update human prompt with improvement request and current instructions."""
    # Convert current_instructions to string format for the prompt if it's JSON
    if isinstance(current_instructions, dict):
        instructions_text = json.dumps(current_instructions, indent=2)
    else:
        instructions_text = str(current_instructions)
    
    return INCREMENTAL_UPDATE_HUMAN_PROMPT_TEMPLATE.format(
        improvement_request=improvement_request,
        current_instructions=instructions_text
    )

def get_incremental_agent_update_system_prompt(used_blocks: list, example: str) -> str:
    """Get the incremental agent update system prompt with used blocks and example."""
    return INCREMENTAL_AGENT_UPDATE_SYSTEM_PROMPT_TEMPLATE.format(
        used_blocks=json.dumps(used_blocks, indent=2),
        example=example
    )

def get_incremental_agent_update_human_prompt(current_agent_json: dict, updated_instructions: str) -> str:
    """Get the incremental agent update human prompt with current agent JSON and updated instructions."""
    return INCREMENTAL_AGENT_UPDATE_HUMAN_PROMPT_TEMPLATE.format(
        current_agent_json=json.dumps(current_agent_json, indent=2),
        updated_instructions=updated_instructions
    )

def get_patch_generation_system_prompt(block_summaries: list) -> str:
    """Get the patch generation system prompt with block summaries."""
    return PATCH_GENERATION_SYSTEM_PROMPT_TEMPLATE.format(
        block_summaries=json.dumps(block_summaries, indent=2)
    )

def get_patch_generation_human_prompt(agent_summary: dict, current_agent: dict, update_request: str) -> str:
    """Get the patch generation human prompt with agent context and update request."""
    return PATCH_GENERATION_HUMAN_PROMPT_TEMPLATE.format(
        agent_summary=json.dumps(agent_summary, indent=2),
        current_agent=json.dumps(current_agent, indent=2),
        update_request=update_request
    )


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


