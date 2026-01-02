import json
import os
import pickle
from pathlib import Path

import torch
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    SEMANTIC_THRESHOLD,
    SPLITTER_TYPE,
    SPLITTING_MODEL,
)
from dotenv import dotenv_values
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import get_client

config = dotenv_values(".env")

OPENAI_API_KEY = config.get("OPENAI_API_KEY")

os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

# Initialize Langfuse client (reads from environment variables)
langfuse = get_client()

torch.cuda.is_available()
torch.set_float32_matmul_precision("high")

# Create a span for the embedding pipeline
with langfuse.start_as_current_observation(
    as_type="span", name="embedding_pipeline"
) as embedding_span:
    splitter_config = {
        "splitter_type": SPLITTER_TYPE,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
    }

    # Initialize text splitter based on config
    if SPLITTER_TYPE == "semantic":
        splitting_embeddings = HuggingFaceEmbeddings(
            model_name=SPLITTING_MODEL,
            model_kwargs={"device": "cuda"},
        )

        text_splitter = SemanticChunker(
            embeddings=splitting_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=SEMANTIC_THRESHOLD,
        )

        splitter_config.update(
            {
                "splitting_model": SPLITTING_MODEL,
                "threshold": SEMANTIC_THRESHOLD,
            }
        )

        print(
            f"Using SemanticChunker with {SPLITTING_MODEL}, threshold={SEMANTIC_THRESHOLD}"
        )

    elif SPLITTER_TYPE == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        splitter_config.update(
            {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
        )

        print(
            f"Using RecursiveCharacterTextSplitter with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
        )

    else:
        raise ValueError(
            f"Unknown splitter type: {SPLITTER_TYPE}. Use 'semantic' or 'recursive'"
        )

    embedding_span.update(metadata=splitter_config)

    # Load documents
    docs_dir = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/cleaned_texts")
    documents = []

    for dir in docs_dir.glob("*"):
        txt_file = Path(dir).glob("LLM_clean_text*.txt")
        txt_files = list(txt_file)
        if not txt_files:
            continue

        loader = TextLoader(txt_files[0], encoding="utf-8")
        loaded_docs = loader.load()

        json_file = Path(f"{dir}/meta_data.json")
        if not json_file.exists():
            continue

        with json_file.open("r", encoding="utf-8") as f:
            meta_from_json = json.load(f)

        for doc in loaded_docs:
            doc.metadata["title"] = meta_from_json["title"]
            doc.metadata["language"] = meta_from_json["language"]
            doc.metadata["url"] = meta_from_json["url"]
            doc.metadata["urldate"] = meta_from_json["urldate"]

        documents.extend(loaded_docs)

    print(f"Loaded {len(documents)} documents")

    # Splitting span
    with langfuse.start_as_current_observation(
        as_type="span", name="text_splitting"
    ) as split_span:
        split_docs = text_splitter.split_documents(documents)

        # Calculate and log chunk statistics
        chunk_lengths = [len(doc.page_content) for doc in split_docs]
        chunk_stats = {
            "total_chunks": len(split_docs),
            "total_documents": len(documents),
            "avg_chunk_size": int(sum(chunk_lengths) / len(chunk_lengths)),
            "max_chunk_size": max(chunk_lengths),
            "min_chunk_size": min(chunk_lengths),
            "median_chunk_size": int(sorted(chunk_lengths)[len(chunk_lengths) // 2]),
        }

        # Calculate percentiles for better understanding
        sorted_lengths = sorted(chunk_lengths)
        chunk_stats["p25_chunk_size"] = sorted_lengths[len(sorted_lengths) // 4]
        chunk_stats["p75_chunk_size"] = sorted_lengths[3 * len(sorted_lengths) // 4]
        chunk_stats["p95_chunk_size"] = sorted_lengths[int(0.95 * len(sorted_lengths))]

        print("\n" + "=" * 80)
        print("CHUNK STATISTICS:")
        print(f"Total documents: {chunk_stats['total_documents']}")
        print(f"Total chunks: {chunk_stats['total_chunks']}")
        print(
            f"Chunks per document: {chunk_stats['total_chunks'] / chunk_stats['total_documents']:.1f}"
        )
        print(f"\nChunk sizes (characters):")
        print(f"  Min:    {chunk_stats['min_chunk_size']:>6}")
        print(f"  25th %: {chunk_stats['p25_chunk_size']:>6}")
        print(f"  Median: {chunk_stats['median_chunk_size']:>6}")
        print(f"  Average:{chunk_stats['avg_chunk_size']:>6}")
        print(f"  75th %: {chunk_stats['p75_chunk_size']:>6}")
        print(f"  95th %: {chunk_stats['p95_chunk_size']:>6}")
        print(f"  Max:    {chunk_stats['max_chunk_size']:>6}")
        print("=" * 80 + "\n")

        split_span.update(output=chunk_stats)

    # Save split documents
    splitter_suffix = f"{SPLITTER_TYPE}"
    if SPLITTER_TYPE == "semantic":
        splitter_suffix += f"_{SPLITTING_MODEL.split('/')[-1]}_t{SEMANTIC_THRESHOLD}"
    else:
        splitter_suffix += f"_s{CHUNK_SIZE}_o{CHUNK_OVERLAP}"

    splits_file = Path(
        f"/home/bartek/Kod/PD/praca_dyplomowa/dane/vectordb/{splitter_suffix}_split_documents.pkl"
    )
    with open(splits_file, "wb") as f:
        pickle.dump(split_docs, f)

    print(f"Saved splits to: {splits_file}")

    # Clean up GPU memory if semantic splitter was used
    if SPLITTER_TYPE == "semantic":
        del splitting_embeddings
        del text_splitter
        torch.cuda.empty_cache()
        print("emptied CUDA cache")

    # Embedding span
    with langfuse.start_as_current_observation(
        as_type="span", name="embedding_creation"
    ) as embed_span:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )

        chroma_db_path = Path(
            f"/home/bartek/Kod/PD/praca_dyplomowa/dane/vectordb/{splitter_suffix}_{EMBEDDING_MODEL.split('/')[-1]}_chroma_db"
        )

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(chroma_db_path),
        )

        print(f"Created vectorstore at: {chroma_db_path}")
        embed_span.update(output={"vectorstore_path": str(chroma_db_path)})

    embedding_span.update(
        output={
            "status": "complete",
            "splits_file": str(splits_file),
            "vectorstore_path": str(chroma_db_path),
            **chunk_stats,
        }
    )

# Flush to ensure all data is sent
langfuse.flush()
print("\nEmbedding pipeline complete!")
