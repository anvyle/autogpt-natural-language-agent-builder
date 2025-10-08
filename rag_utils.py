import asyncio
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from utils import load_json_async

# Import centralized config for secrets management
import config

# Environment is set up automatically by config module

# Configuration
examples_dir = "./examples"
persist_directory = "./chroma_db"
agent_collection_name = "agents_v1"
agent_file = "./data/agent_examples.json"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def agent_to_document(agent):
    name = agent.get("name", "")
    description = agent.get("description", "")
    categories = ", ".join(agent.get("categories", [])) 
    combined = f"{name}\n{description}\nCategories: {categories}" if categories else f"{name}\n{description}"
    return Document(
        page_content=combined,
        metadata={
            "id": agent.get("id", ""),
            "name": name,
            "categories": categories,
        },
    )

async def build_agent_vector_store() -> VectorStore:
    logging.info("🛠️ Building Chroma vector store for agents...")
    agents = await load_json_async(agent_file)
    documents = [agent_to_document(agent) for agent in agents]
    store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=agent_collection_name,
        persist_directory=persist_directory,
    )
    logging.info("✅ Chroma vectorstore built and persisted.")
    return store

async def query_agent_store(query: str, k: int = 3):
    full_dataset = await load_json_async(agent_file)
    store = Chroma(
        embedding_function=embedding_model,
        collection_name=agent_collection_name,
        persist_directory=persist_directory,
    )
    results = store.similarity_search(query, k=k)
    matched_example_names = [doc.metadata.get("name", "") for doc in results]
    matched_examples = [
        entry for entry in full_dataset if entry.get("name", "") in matched_example_names
    ]
    
    for i, doc in enumerate(results):
    #     print(f"\nResult {i+1}:")
        print(f"Metadata: {doc.metadata}")
    #     print(f"Content: {doc.page_content}")
    return matched_examples

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        await build_agent_vector_store()
        await query_agent_store("send an email to the user")

    asyncio.run(main())
