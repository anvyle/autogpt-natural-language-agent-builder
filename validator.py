import os
import json
import logging
from dotenv import load_dotenv
from pydantic import ValidationError

from backend.data.integrations import Webhook
from backend.data.graph import GraphModel

load_dotenv()

USER_ID = os.getenv("USER_ID")

def validate_agent_json(agent_json, for_run: bool = False) -> tuple[bool, str]:
    logging.info("🔍 Validating agent JSON...")
    try:
        agent_json["user_id"] = USER_ID
        agent_json["nodes"] = [
            {**node, "graph_id": agent_json["id"], "graph_version": agent_json.get("version", 1)}
            for node in agent_json["nodes"]
        ]
        data = json.loads(agent_json) if isinstance(agent_json, str) else agent_json
        graph = GraphModel(**data)
        graph.validate_graph(for_run)
        logging.info("✅ Agent JSON validation successful.")
        return True, "OK"
    except (ValidationError, ValueError) as err:
        return False, str(err)
    