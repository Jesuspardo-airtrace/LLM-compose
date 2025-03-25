from praisonaiagents import Agent

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "praison",
            "path": ".praison"
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "deepseek-r1:latest",
            "temperature": 0,
            "max_tokens": 8000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 1536
        },
    },
}

agent = Agent(
    name="Knowledge Agent",
    instructions="You answer questions based on the provided knowledge.",
    knowledge=["app/DATA/Análisis de Series Temporales (Avanzado).pdf"], # Indexing
    knowledge_config=config,
    user_id="user1",
    llm="deepseek-r1"
)

agent.start("Dame un ejemplo de un análisis de datos de una EDAR") # Retrieval
