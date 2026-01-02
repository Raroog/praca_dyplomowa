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
v = "v0.0"
if config:
    SPLITTING_MODEL = config.get(f"SPLITTING_MODEL{v}", "")
    CHROMA_DIR = Path(config.get(f"CHROMA_DIR{v}"))
    SPLITS_FILE = Path(config.get(f"SPLITS_FILE{v}"))
    COLLECTION_NAME = config.get("COLLECTION_NAME")

    EMBEDDING_MODEL = config.get(f"EMBEDDING_MODEL{v}", "")
    RERANKER_MODEL = config.get(f"RERANKER_MODEL{v}")
    OLLAMA_MODEL = config.get("OLLAMA_MODEL")

    DENSE_K = int(config.get("DENSE_K"))
    SPARSE_K = int(config.get("SPARSE_K"))
    RERANK_TOP_N = int(config.get("RERANK_TOP_N"))
    DENSE_WEIGHT = float(config.get("DENSE_WEIGHT"))
    TEMPERATURE = float(config.get("TEMPERATURE"))
