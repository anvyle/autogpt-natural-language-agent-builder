```git clone https://github.com/Significant-Gravitas/AutoGPT-Agent-Generator.git```

```cd AutoGPT-Agent-Generator```

```cp .env.example .env```

Please set up ``GOOGLE_API_KEY`` and ``USER_ID`` in .env

```python3 -m venv .venv```

```source .venv/bin/activate```

```git clone https://github.com/Significant-Gravitas/AutoGPT.git```

```cd AutoGPT```

```cp autogpt_platform/backend/.env.default autogpt_platform/backend/.env```

```cd autogpt_platform/backend```

```docker compose -f docker-compose.test.yaml up -d```

```poetry install```

```poetry run prisma generate```

```poetry run prisma migrate deploy```

```cd ../../../```

```cp -r AutoGPT/autogpt_platform/backend backend```

```pip install -r requirements.txt```

```python3 rag_utils.py```

```streamlit run streamlit_agent_builder.py```
