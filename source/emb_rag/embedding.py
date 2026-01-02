import json
import pickle
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from .config import COLLECTION_NAME, EMBEDDING_MODEL, SPLITTING_MODEL

torch.cuda.is_available()
torch.set_float32_matmul_precision("high")

splitting_embeddings = HuggingFaceEmbeddings(
    model_name=SPLITTING_MODEL,
    model_kwargs={
        "device": "cuda",
    },
)

text_splitter = SemanticChunker(
    embeddings=splitting_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=0.9,
)

docs_dir = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/cleaned_texts")
documents = []

for dir in docs_dir.glob("*"):
    txt_file = Path(dir).glob("LLM_clean_text*.txt")
    loader = TextLoader(list(txt_file)[0], encoding="utf-8")
    loaded_docs = loader.load()

    json_file = Path(f"{dir}/meta_data.json")
    with json_file.open("r", encoding="utf-8") as f:
        meta_from_json = json.load(f)

    for doc in loaded_docs:
        doc.metadata["title"] = meta_from_json["title"]
        doc.metadata["language"] = meta_from_json["language"]
        doc.metadata["url"] = meta_from_json["url"]
        doc.metadata["urldate"] = meta_from_json["urldate"]

    documents.extend(loaded_docs)

print(f"Loaded {len(documents)} documents")

split_docs = text_splitter.split_documents(documents)

splits_file = Path(
    f"/home/bartek/Kod/PD/praca_dyplomowa/dane/vectordb/{SPLITTING_MODEL.split('/')[1]}_split_documents.pkl"
)
with open(splits_file, "wb") as f:
    pickle.dump(split_docs, f)

del splitting_embeddings
del text_splitter
torch.cuda.empty_cache()

with open(splits_file, "rb") as f:
    split_docs = pickle.load(f)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
)
chroma_db_path = f"/home/bartek/Kod/PD/praca_dyplomowa/dane/vectordb/{EMBEDDING_MODEL.split('/')[1]}_chroma_db"
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory="./chroma_db",
)
