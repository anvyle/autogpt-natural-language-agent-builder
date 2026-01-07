# AutoGPT Agent Generator

🤖 A powerful tool for building AI agents from natural language descriptions. Choose between a conversational Streamlit interface or a RESTful API for programmatic access.

## Features

- 📋 **Task Decomposition**: Break down goals into step-by-step instructions
- 🤖 **Agent Generation**: Generate complete agent JSON from instructions
- 🔄 **Patch-Based Updates**: Surgically modify existing agents using minimal patches that preserve unchanged parts
- 📝 **Template Modification**: Modify existing agent templates using the same patch-based system
- ❓ **Interactive Clarification**: Handle clarifying questions for ambiguous requests

## Two Interfaces

### 1. Streamlit Web Interface (Interactive)
A ChatGPT-style conversational interface for building agents interactively.

### 2. FastAPI RESTful API (Programmatic)
A RESTful API for integrating agent generation into your applications.

## Quick Start

### Prerequisites

- Python 3.8+
- Google API Key (for Gemini model)
- AutoGPT Platform API Key (for fetching blocks)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Significant-Gravitas/AutoGPT-Agent-Generator.git
   cd AutoGPT-Agent-Generator
   ```

2. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   # Google Generative AI API Key (required for agent generation)
   GOOGLE_API_KEY=your_google_api_key_here
   
   # AutoGPT Platform API Key (required for fetching blocks)
   # Get your API key from: https://platform.agpt.co/settings/api-keys
   AUTOGPT_API_KEY=your_autogpt_api_key_here
   
   # Optional: AutoGPT Blocks API URL (defaults to v1 endpoint)
   # AUTOGPT_BLOCKS_API_URL=https://backend.agpt.co/external-api/v1/blocks
   ```
   
   **Getting your AutoGPT API Key:**
   1. Go to [AutoGPT Platform](https://platform.agpt.co)
   2. Click the top-right menu → Settings
   3. Navigate to API Keys
   4. Click "Create Key"

3. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Streamlit Interface (Recommended for Interactive Use)

Run the Streamlit web interface:

```bash
streamlit run streamlit_agent_builder.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Conversational chat interface
- Auto mode for automatic workflow execution
- Manual mode for step-by-step review
- Create new agents or modify templates
- Download generated agents as JSON

### Option 2: FastAPI Server (Recommended for Programmatic Use)

#### Start the API Server

```bash
python fastapi_server.py
```

Or using uvicorn directly:

```bash
uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

#### Test the API

Use the interactive Swagger UI at `http://localhost:8000/docs` to test endpoints manually, or use curl/Postman to make requests.

See [API_README.md](API_README.md) for complete API documentation.

## Project Structure

```
AutoGPT-Agent-Generator/
├── fastapi_server.py              # FastAPI server implementation
├── streamlit_agent_builder.py     # Streamlit web interface
├── agent_builder.py               # Core agent generation logic
├── blocks_fetcher.py              # Dynamic blocks fetching from API
├── config.py                      # Configuration and secrets management
├── utils.py                       # Utility functions
├── validator.py                   # Agent validation
├── rag_utils.py                   # RAG utilities
├── data/                          # Block definitions and cache
│   ├── blocks_cache.json         # Cached blocks from API
│   ├── blocks_2025_11_11_edited.json    # Fallback blocks
│   └── Resume_Rater_AI.json      # Example agent
├── generated_agents/              # Output directory for agents
│   └── YYYYMMDD/                 # Date-organized agents
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Docker image definition
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── API_README.md                  # Detailed API documentation
```

## API Endpoints

### Core Endpoints

- `POST /api/decompose-description` - Decompose goal into steps
- `POST /api/generate-agent` - Generate agent from instructions
- `POST /api/update-agent` - Update existing agent using patch-based incremental updates
- `POST /api/template-modification` - Modify template agent using patch-based updates
- `GET /api/blocks` - Get available blocks
- `GET /health` - Health check

See [API_README.md](API_README.md) for detailed documentation.

## Development Workflow

### Creating a New Agent (Streamlit)

1. Launch Streamlit interface: `streamlit run streamlit_agent_builder.py`
2. Select "Create New Agent"
3. Describe your goal
4. Answer any clarifying questions (if needed)
5. Review step-by-step instructions
6. Generate and download your agent

### Creating a New Agent (API)

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Decompose goal into steps
goal = "Create a weather report agent"
response = requests.post(
    f"{BASE_URL}/api/decompose-description",
    json={"description": goal}
)
decomposition = response.json()

# 2. Generate agent
if decomposition.get("type") == "instructions":
    agent_response = requests.post(
        f"{BASE_URL}/api/generate-agent",
        json={"instructions": decomposition}
    )
    agent_json = agent_response.json()["agent_json"]
```

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here
AUTOGPT_API_KEY=your_autogpt_api_key_here

# Optional
AUTOGPT_BLOCKS_API_URL=https://backend.agpt.co/external-api/v1/blocks
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING=true
```

The `config.py` module handles secrets management and supports both local development (`.env`) and cloud deployment (Streamlit Cloud secrets).

### Dynamic Blocks Fetching

The agent generator now fetches blocks dynamically from the AutoGPT platform API at startup:

- **Automatic Updates**: Blocks are always up-to-date with the latest platform changes
- **Smart Caching**: Blocks are cached locally for 24 hours to minimize API calls
- **Fallback Support**: If the API is unavailable, the system falls back to a cached version or hard-coded blocks file
- **Performance**: Large API responses are handled efficiently with streaming and caching

The blocks are fetched once at application startup and cached. To force a refresh:
- Restart the application after 24 hours
- Delete the `data/blocks_cache.json` file and restart
- Use the `force_refresh=True` parameter in `initialize_blocks()` (for custom implementations)

## Testing

### Test the Streamlit Interface

```bash
streamlit run streamlit_agent_builder.py
```

### Test the API Server

```bash
# Start the server
python fastapi_server.py
```

Then visit `http://localhost:8000/docs` to use the interactive Swagger UI for testing endpoints, or use curl/Postman to make API requests.

## Generated Agents

Generated agents are saved to:
```
generated_agents/YYYYMMDD/Agent_Name.json
```

You can import these JSON files directly into your AutoGPT platform.

## Advanced Usage

### Auto Mode (Streamlit)

Enable auto mode in the Streamlit interface for fully automated agent generation. All steps will be processed automatically except clarifying questions.

### Programmatic Integration (API)

Integrate agent generation into your applications:

```python
import requests

# Generate agent from instructions
response = requests.post(
    "http://localhost:8000/api/generate-agent",
    json={
        "instructions": {...}
    }
)

agent_json = response.json()["agent_json"]

# Update existing agent using natural language
update_response = requests.post(
    "http://localhost:8000/api/update-agent",
    json={
        "update_request": "Add error handling to the email step",
        "current_agent_json": agent_json
    }
)

updated_agent = update_response.json()["agent_json"]
```

### Template Modification

Modify existing agent templates using patch-based updates:

**Streamlit:**
1. Select "Modify Template Agent"
2. Upload your template JSON
3. Describe desired modifications (e.g., "Add email notifications after each step")
4. Receive updated agent directly

**API:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/template-modification",
    json={
        "template_agent_json": {...},
        "modification_request": "Add email notifications after each step"
    }
)

modified_agent = response.json()["agent_json"]
```

### Updating Existing Agents

Update existing agents using natural language requests:

**Streamlit:**
1. After generating an agent, click "Improve This Agent"
2. Describe what you want to change (e.g., "Add error handling to step 3")
3. Receive updated agent with changes applied

**API:**
```python
response = requests.post(
    "http://localhost:8000/api/update-agent",
    json={
        "update_request": "Add error handling to the email sending step",
        "current_agent_json": existing_agent_json
    }
)

updated_agent = response.json()["agent_json"]
```

The patch-based system only modifies targeted parts while preserving all unchanged parts exactly.

## Troubleshooting

### Blocks Not Loading

- Ensure `data/blocks_2025_11_11_edited.json` exists
- Check file permissions
- Review logs for errors

### API Key Errors

- Verify `GOOGLE_API_KEY` is set in `.env`
- Check that the key is valid
- Ensure proper permissions

### Connection Errors (API)

- Make sure the API server is running
- Check that port 8000 is not in use
- Verify firewall settings

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- Check the documentation
- Review existing issues on GitHub
- Create a new issue with details

## Documentation

- [API Documentation](API_README.md) - Complete API reference
- [Streamlit Interface Guide](streamlit_agent_builder.py) - Docstrings in code
- [Agent Builder Logic](agent_builder.py) - Core functions documentation

## License

[Add your license information here]

## Acknowledgments

Built for AutoGPT platform by the community.

## Version History

- **v1.0.0** (Current)
  - Initial release with Streamlit interface
  - FastAPI server implementation
  - Complete agent generation workflow
  - **Patch-based incremental updates** - Surgical modifications that preserve unchanged parts
  - **Direct agent updates** - Update agents using natural language without intermediate steps
  - Template modification using patch-based system
