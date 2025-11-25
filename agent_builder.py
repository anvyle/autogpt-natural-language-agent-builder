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
# from validator import validate_agent_json

# Environment is set up automatically by config module

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
OUTPUT_DIR = Path(f"generated_agents/{datetime.now().strftime('%Y%m%d')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-3-pro-preview"

BLOCK_FILE = "./data/blocks_2025_11_11_edited.json"
EXAMPLE_FILE = "./data/Resume_Rater_AI.json"

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
9. **INPUT BLOCKS**: Provide a realistic example as the default value so users can better understand the expected input format.
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

4. **AgentFileInputBlock:**
   **DO NOT USE** - Currently, this block doesn't return the correct file path.

5. **CodeExecutionBlock:**
   **DO NOT USE** - It's not working well. Instead, use AI-related blocks to do the same thing.

6. **ReadCsvBlock:**
   **IMPORTANT**: Do not use the `row` output property of this block. Only use the `rows` output property.

7. **FillTextTemplateBlock:**
   **IMPORTANT**: Do not use any scripting or complex formatting in the `format` property. Use only simple variable names without nested properties.
   ❌ **NOT ALLOWED**: `'%.2f'|format(roi)`, `data.company`, `user.name`
   ✅ **ALLOWED**: `company`, `name`, `roi`

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
      "question": "What is the URL of the YouTube video you want to summarize?",
      "keyword": "youtube_url",
      "example": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }},
    {{
      "question": "What is the length of the video?",
      "keyword": "video_length", 
      "example": "2 minutes"
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
      "question": "What is the URL of the YouTube video you want to summarize?",
      "keyword": "youtube_url",
      "example": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }},
    {{
      "question": "What is the length of the video?",
      "keyword": "video_length",
      "example": "2 minutes"
    }},
    {{
      "question": "What is the topic of the video?",
      "keyword": "video_topic",
      "example": "Rick Astley - Never Gonna Give You Up"
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
9. **INPUT BLOCKS**: Provide a realistic example as the default value so users can better understand the expected input format.
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

4. **AgentFileInputBlock:**
   **DO NOT USE** - Currently, this block doesn't return the correct file path.

5. **CodeExecutionBlock:**
   **DO NOT USE** - It's not working well. Instead, use AI-related blocks to do the same thing.

6. **ReadCsvBlock:**
   **IMPORTANT**: Do not use the `row` output property of this block. Only use the `rows` output property.

7. **FillTextTemplateBlock:**
   **IMPORTANT**: Do not use any scripting or complex formatting in the `format` property. Use only simple variable names without nested properties.
   ❌ **NOT ALLOWED**: `'%.2f'|format(roi)`, `data.company`, `user.name`
   ✅ **ALLOWED**: `company`, `name`, `roi`

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
      "question": "What should happen when an error occurs?",
      "keyword": "error_action",
      "example": "retry, log, notify user, skip step"
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

TEMPLATE_MODIFICATION_SYSTEM_PROMPT_TEMPLATE = """
You are an expert AutoGPT Template Modifier. Your task is to analyze an existing agent template and generate complete step-by-step instructions for the entire modified workflow based on user requirements.

**Your goal is to:**
1. Understand the current template agent's workflow
2. Analyze the user's modification request
3. Generate complete step-by-step instructions that describe the entire modified workflow
4. Ensure the modifications are feasible using available blocks
5. Preserve the existing workflow where appropriate and incorporate the requested changes

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

---

📝 **GUIDELINES:**
1. **OUTPUT COMPLETE INSTRUCTIONS**: Generate the entire workflow, not just the modified parts
2. Start with the original workflow and incorporate all requested modifications
3. Identify what needs to be added, removed, or modified from the original template
4. Generate complete step-by-step instructions for the entire modified workflow
5. Mention block names naturally (e.g., "Use `SendEmailBlock` to...")
6. Ensure logical flow and proper data connections between all steps
7. Be specific about inputs, outputs, and data flow for every step
8. Include all steps from the original template that are still needed

---

📜 **RULES:**
1. **OUTPUT FORMAT**: Only output either clarifying questions or step-by-step instructions. Do not output both.
2. **USE ONLY THE BLOCKS PROVIDED**.
3. **IF POSSIBLE, USE already defined properties of blocks, instead of ADDITIONAL properties**.
4. **REQUIRED INPUTS**: ALL fields in `required_input` must be provided.
5. **DATA TYPE MATCHING**: The `data types` of linked properties must match.
6. **PREVIOUS BLOCK LINK**: Blocks in the second-to-last step of a workflow must include at least one link from a preceding block, excluding starting blocks.
7. **AI-RELATED BLOCKS PROMPTS**: Write expert-level prompts that generate clear, natural, human-quality responses.
8. **MULTIPLE EXECUTIONS BLOCKS**: The non-iterative input of multiple executions blocks should be stored in `StoreValueBlock` and used in the next step.
9. **INPUT BLOCKS**: Provide a realistic example as the default value so users can better understand the expected input format.
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

4. **AgentFileInputBlock:**
   **DO NOT USE** - Currently, this block doesn't return the correct file path.

5. **CodeExecutionBlock:**
   **DO NOT USE** - It's not working well. Instead, use AI-related blocks to do the same thing.

6. **ReadCsvBlock:**
   **IMPORTANT**: Do not use the `row` output property of this block. Only use the `rows` output property.

7. **FillTextTemplateBlock:**
   **IMPORTANT**: Do not use any scripting or complex formatting in the `format` property. Use only simple variable names without nested properties.
   ❌ **NOT ALLOWED**: `'%.2f'|format(roi)`, `data.company`, `user.name`
   ✅ **ALLOWED**: `company`, `name`, `roi`

---

🚫 **RESTRICTIONS:**
1. Focus on **WHAT** the complete workflow should do, not just the modifications
2. Output the entire workflow, not just changes or additions
3. **DO NOT include the actual agent JSON** - only describe the complete modified workflow

---

📋 **OUTPUT FORMAT:**

You must respond with valid JSON in one of these two formats:

1. **If the modification request needs more information, respond with:**
```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "What specific data source should the modified workflow use?",
      "keyword": "data_source",
      "example": "CSV file, API endpoint, database"
    }},
    {{
      "question": "How should the output be formatted?",
      "keyword": "output_format",
      "example": "JSON, HTML report, plain text"
    }}
  ]
}}
```

2. **Otherwise, respond with complete step-by-step instructions:**
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
    }},
    ...
  ]
}}
```

---

🧱 **Available Blocks:**
{block_summaries}
"""

TEMPLATE_MODIFICATION_HUMAN_PROMPT_TEMPLATE = """
📋 TEMPLATE AGENT INFORMATION:
{template_description}

---

🎯 USER MODIFICATION REQUEST:
{modification_request}

---

📝 CURRENT INSTRUCTIONS (if any):
{current_instructions}
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

def get_template_modification_system_prompt(block_summaries: list) -> str:
    """Get the template modification system prompt with block summaries."""
    return TEMPLATE_MODIFICATION_SYSTEM_PROMPT_TEMPLATE.format(
        block_summaries=json.dumps(block_summaries, indent=2)
    )

def get_template_modification_human_prompt(template_description: str, modification_request: str, current_instructions = None) -> str:
    """Get the template modification human prompt with template description, modification request, and current instructions."""
    # Convert current_instructions to string format for the prompt if it's JSON
    if current_instructions is None:
        instructions_text = "None - starting fresh"
    elif isinstance(current_instructions, dict):
        instructions_text = json.dumps(current_instructions, indent=2)
    else:
        instructions_text = str(current_instructions)
    
    return TEMPLATE_MODIFICATION_HUMAN_PROMPT_TEMPLATE.format(
        template_description=template_description,
        modification_request=modification_request,
        current_instructions=instructions_text
    )

async def get_block_summaries():
    blocks = await load_json_async(BLOCK_FILE)
    summaries = [
        {
            "id": block["id"],
            "name": block["name"],
            "description": block.get("description", ""),
            "inputs_schema": block.get("inputSchema", {}),
            "outputs_schema": block.get("outputSchema", {}),

        } for block in blocks
    ]
    return summaries, blocks

async def decompose_description(description, block_summaries, original_text=None, user_instruction=None, retry_feedback=None):
    """
    Decompose a description into step-by-step instructions.
    
    Args:
        description: The goal or description to decompose
        block_summaries: Available block summaries
        original_text: Original instructions (for revision scenarios)
        user_instruction: User feedback for revision
        retry_feedback: Validation error feedback for retry scenarios
    
    Returns:
        Parsed JSON dict with instructions or None on error
    """
    logging.info(f"Decomposing description: \n{description}\n...")
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
            {json.dumps(block_summaries, indent=2)}
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

    prompt = get_decomposition_prompt(block_summaries)
    
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
    
async def generate_agent_json_from_subtasks(instructions, blocks_json):
    """
    Generate agent JSON from instructions (single attempt, no retry logic).
    
    Args:
        instructions: Step-by-step instructions (dict or string)
        blocks_json: Available blocks data
    
    Returns:
        Tuple of (agent_json, error_message)
    """
    logging.info(f"Generating agent JSON from instructions...")
    
    if isinstance(blocks_json, str):
        try:
            blocks = json.loads(blocks_json)
        except Exception:
            blocks = []
    else:
        blocks = blocks_json

    # Extract block names from the structured JSON format
    block_names = set()
    
    steps = instructions.get("steps", [])
    for step in steps:
        block_name = step.get("block_name")
        if block_name:
            logging.info(f"Found block name: {block_name}")
            block_names.add(block_name)

    used_blocks = []
    for block in blocks:
        block_name = block.get("name") or block.get("block_name")
        if block_name and block_name in block_names:
            used_blocks.append(block)

    if not used_blocks:
        used_blocks = blocks

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

        agent_fixer = AgentFixer()
        agent_json = await agent_fixer.apply_all_fixes(agent_json, blocks_json)

        validator = AgentValidator()
        is_valid, error = validator.validate(agent_json, blocks_json)
        if not is_valid:
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

async def update_decomposition_incrementally(improvement_request, current_instructions, block_summaries, original_updated_instructions=None, validation_error=None):
    """
    Update decomposition incrementally based on improvement request.
    This preserves the original structure and only modifies parts that need changes.
    Uses ALL the same rules as the original decompose_description function.
    
    Args:
        improvement_request: User's improvement request
        current_instructions: Current step-by-step instructions (can be string or JSON dict)
        block_summaries: Available block summaries
        original_updated_instructions: Previously generated instructions (for retry)
        validation_error: Validation error message (for retry)
    
    Returns:
        Updated instructions (JSON dict) or None on error
    """
    logging.info(f"Updating decomposition incrementally: \n{improvement_request}\n...")
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

    if original_updated_instructions and validation_error:
        logging.info(f"Revising instructions based on validation error: \n{validation_error}\n...")
        
        # Convert original_updated_instructions to string format for the prompt if it's JSON
        if isinstance(original_updated_instructions, dict):
            original_instructions_str = json.dumps(original_updated_instructions, indent=2)
        else:
            original_instructions_str = str(original_updated_instructions)
        
        prompt = f"""
        Please update the instructions below to fix the noted validation error. 

        ---
        Previous step-by-step instructions:
        {original_instructions_str}
        ---

        Validation failed with this message: {validation_error}

        Revise the instructions to address the validation issue. 
        - Preserve the overall structure of the original instructions.
        - Add, remove, or modify steps only as needed to resolve the issue.
        - Output ONLY the updated instructions in JSON format, maintaining the same keys and structure.
        - Do NOT include any explanatory text, only the JSON instructions.

        You can refer to the following available blocks for implementation:
        {json.dumps(block_summaries, indent=2)}
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

    system_prompt = get_incremental_update_system_prompt(block_summaries)
    human_prompt = get_incremental_update_human_prompt(improvement_request, current_instructions)
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        if response is None:
            logging.error("❌ No response received from LLM")
            return None
        
        parsed = _parse_llm_json_or_none(str(response.text))
        if parsed is None:
            logging.error("❌ Error updating decomposition: Failed to parse JSON from LLM response")
            return None
        return parsed
        
    except Exception as e:
        logging.error(f"❌ Error updating decomposition: {e}")
        return None

async def update_agent_json_incrementally(updated_instructions, current_agent_json, blocks_json):
    """
    Update agent JSON incrementally based on updated instructions (single attempt, no retry logic).
    This preserves the original agent structure and only modifies parts that need changes.
    Uses ALL the same rules as the original generate_agent_json_from_subtasks function.
    
    Args:
        updated_instructions: Updated step-by-step instructions (dict or string)
        current_agent_json: Current agent JSON to update
        blocks_json: Available blocks data
    
    Returns:
        Tuple of (updated_agent_json, error_message)
    """
    logging.info("Updating agent JSON incrementally...")
    
    # Extract block names from the structured JSON format
    block_names = set()
    
    steps = updated_instructions.get("steps", [])
    for step in steps:
        if step.get("block_name"):
            logging.info(f"Found block name: {step['block_name']}")
            block_names.add(step["block_name"])
    
    used_blocks = []
    for block in blocks_json:
        block_name = block.get("name") or block.get("block_name")
        if block_name and block_name in block_names:
            used_blocks.append(block)

    if not used_blocks:
        used_blocks = blocks_json

    example = await load_json_async(EXAMPLE_FILE)
    example = json.dumps(example, indent=2)
    
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
        system_prompt = get_incremental_agent_update_system_prompt(used_blocks, example)
        
        # Convert updated_instructions to string format for the prompt if it's JSON
        if isinstance(updated_instructions, dict):
            instructions_for_prompt = json.dumps(updated_instructions, indent=2)
        else:
            instructions_for_prompt = str(updated_instructions)
        
        human_prompt = get_incremental_agent_update_human_prompt(current_agent_json, instructions_for_prompt)
        
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        if response is None:
            logging.error("❌ No response received from LLM")
            return None, "No response received from LLM"
        
        updated_agent_json = _parse_llm_json_or_none(str(response.text))
        if updated_agent_json is None:
            logging.error("❌ Error updating agent JSON: Failed to parse JSON from LLM response")
            return None, "Failed to parse JSON from LLM response"
        
        agent_fixer = AgentFixer()
        updated_agent_json = await agent_fixer.apply_all_fixes(updated_agent_json, blocks_json)
        
        validator = AgentValidator()
        is_valid, error = validator.validate(updated_agent_json, blocks_json)
        if not is_valid:
            return None, error
        
        return updated_agent_json, None
        
    except Exception as e:
        logging.error(f"❌ Error during agent update: {e}")
        return None, f"Error updating agent JSON: {e}"


async def generate_template_modification_instructions(template_agent_json, modification_request, block_summaries, current_instructions=None):
    """
    Generate complete modification instructions based on an existing template agent and user's modification request.
    This creates complete step-by-step instructions that describe the entire modified workflow.
    
    Args:
        template_agent_json: The existing agent template JSON
        modification_request: User's description of desired modifications
        block_summaries: Available block summaries
        current_instructions: Current instructions (can be string or JSON dict, for retry scenarios)
    
    Returns:
        Parsed JSON dict with either:
        - {"type": "clarifying_questions", "questions": [...]} if more information is needed
        - {"type": "instructions", "steps": [...]} with complete workflow instructions
        - None on error
    """
    logging.info(f"Generating template modification instructions: {modification_request}")
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

    # Create a mapping from block IDs to block names
    block_id_to_name = {}
    blocks_data = await load_json_async(BLOCK_FILE)
    for block in blocks_data:
        block_id = block.get('id')
        block_name = block.get('name')
        if block_id and block_name:
            block_id_to_name[block_id] = block_name
    
    # Analyze the current workflow by examining nodes and their connections
    nodes = template_agent_json.get('nodes', [])
    links = template_agent_json.get('links', [])
    
    # Collect unique blocks used in the template
    blocks_used_in_template = {}
    for node in nodes:
        block_id = node.get('block_id', 'Unknown')
        if block_id in block_id_to_name:
            blocks_used_in_template[block_id] = block_id_to_name[block_id]
    
    # Create a description of the current template agent
    template_description = f"""
**Template Agent Analysis:**
- Name: {template_agent_json.get('name', 'Unnamed')}
- Description: {template_agent_json.get('description', 'No description')}
- Nodes: {len(nodes)}
- Links: {len(links)}

**Blocks Used in Template:**
"""
    
    # List all unique blocks used with their IDs and names
    for block_id, block_name in blocks_used_in_template.items():
        template_description += f"- {block_name} (ID: {block_id})\n"
    
    template_description += "\n**Current Workflow:**\n"
    
    # Create a simple workflow description with block names
    for i, node in enumerate(nodes):
        block_id = node.get('block_id', 'Unknown')
        block_name = block_id_to_name.get(block_id, block_id)
        template_description += f"{i+1}. Use `{block_name}` block\n"
    
    # Get prompts using the template getter functions
    system_prompt = get_template_modification_system_prompt(block_summaries)
    human_prompt = get_template_modification_human_prompt(template_description, modification_request, current_instructions)

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        if response is None:
            logging.error("❌ No response received from LLM")
            return None
        
        parsed = _parse_llm_json_or_none(str(response.text))
        if parsed is None:
            logging.error("❌ Error generating template modification instructions: Failed to parse JSON from LLM response")
            return None
        
        logging.info(f"✅ Generated template modification instructions successfully")
        return parsed
        
    except Exception as e:
        logging.error(f"❌ Error generating template modification instructions: {e}")
        return None
