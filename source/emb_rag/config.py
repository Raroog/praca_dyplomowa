from pathlib import Path

import yaml


def load_config(config_path="/home/bartek/Kod/PD/praca_dyplomowa/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


config = load_config().get("rag")
v = "v0.1"
if config:
    LANGFUSE_SECRET_KEY = config.get("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = config.get("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST = config.get("LANGFUSE_HOST")

    # Splitter configuration
    SPLITTER_TYPE = config.get(
        "SPLITTER_TYPE", "recursive"
    )  # "semantic" or "recursive"
    SPLITTING_MODEL = config.get(f"SPLITTING_MODEL{v}", "")
    SEMANTIC_THRESHOLD = float(config.get(f"SEMANTIC_THRESHOLD{v}", 0.6))
    CHUNK_SIZE = int(config.get(f"CHUNK_SIZE{v}", 1024))
    CHUNK_OVERLAP = int(config.get(f"CHUNK_OVERLAP{v}", 150))

    # Storage paths
    CHROMA_DIR = Path(config.get(f"CHROMA_DIR{v}"))
    SPLITS_FILE = Path(config.get(f"SPLITS_FILE{v}"))
    COLLECTION_NAME = config.get("COLLECTION_NAME")

    # Model configuration
    EMBEDDING_MODEL = config.get(f"EMBEDDING_MODEL{v}", "")
    RERANKER_MODEL = config.get(f"RERANKER_MODEL{v}")
    OLLAMA_MODEL = config.get("OLLAMA_MODEL")

    # Retrieval configuration
    DENSE_K = int(config.get("DENSE_K"))
    SPARSE_K = int(config.get("SPARSE_K"))
    RERANK_TOP_N = int(config.get("RERANK_TOP_N"))
    DENSE_WEIGHT = float(config.get("DENSE_WEIGHT"))
    TEMPERATURE = float(config.get("TEMPERATURE"))
