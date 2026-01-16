import logging
import os
import pickle
import re
from pathlib import Path

import torch
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DENSE_K,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    OLLAMA_MODEL,
    RELEVANCE_THRESHOLD,
    RERANK_TOP_N,
    RERANKER_MODEL,
    SPARSE_K,
    SPLITS_FILE,
    TEMPERATURE,
)
from dotenv import dotenv_values
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

config = dotenv_values(".env")
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

langfuse = get_client()

ADAPTIVE_PROMPT = """You are a cybersecurity expert.

First, assess if the provided context is relevant to the question.
- If relevant: answer using the context and cite sources
- If not relevant or only partially relevant: answer based on your knowledge, noting what the context does/doesn't cover

Context:
{context}

Question: {question}"""


def load_split_documents(splits_path: Path) -> list[Document]:
    logger.info("Loading split documents from %s", splits_path)
    with splits_path.open("rb") as f:
        return pickle.load(f)


def create_dense_retriever(
    persist_dir: Path,
    collection_name: str,
    model_name: str = EMBEDDING_MODEL,
    k: int = DENSE_K,
):
    logger.info("Loading Chroma vectorstore from %s", persist_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    logger.info("Vectorstore loaded with %d documents", vectorstore._collection.count())
    return vectorstore.as_retriever(search_kwargs={"k": k})


def create_bm25_retriever(
    documents: list[Document], k: int = SPARSE_K
) -> BM25Retriever:
    logger.info("Creating BM25 retriever from %d documents", len(documents))
    return BM25Retriever.from_documents(
        documents,
        k=k,
        preprocess_func=lambda text: re.findall(r"\b\w+\b", text.lower()),
    )


def create_hybrid_retriever(
    dense_retriever,
    sparse_retriever,
    weights: tuple[float, float] = (DENSE_WEIGHT, 1.0 - DENSE_WEIGHT),
) -> EnsembleRetriever:
    logger.info(
        "Creating hybrid retriever with weights: dense=%.2f, sparse=%.2f",
        weights[0],
        weights[1],
    )
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=list(weights),
    )


def rerank_and_filter(
    query: str,
    docs: list[Document],
    cross_encoder: HuggingFaceCrossEncoder,
    top_n: int = RERANK_TOP_N,
    threshold: float = RELEVANCE_THRESHOLD,
) -> tuple[list[Document], list[float]]:
    if not docs:
        return [], []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.score(pairs)

    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    scored = scored[:top_n]
    filtered = [(doc, score) for doc, score in scored if score >= threshold]

    logger.info(
        "Rerank+filter: %d → %d (top_%d) → %d (threshold %.2f)",
        len(docs),
        min(len(docs), top_n),
        top_n,
        len(filtered),
        threshold,
    )

    if not filtered:
        return [], []
    return [doc for doc, _ in filtered], [score for _, score in filtered]


def format_docs(docs: list[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("url", "N/A")
        formatted.append(f"[Source {i}]: {source}\nURL: {url}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def setup_rag(
    chroma_dir: Path = CHROMA_DIR,
    splits_file: Path = SPLITS_FILE,
    collection_name: str = COLLECTION_NAME,
    dense_k: int = DENSE_K,
    sparse_k: int = SPARSE_K,
) -> tuple[EnsembleRetriever, HuggingFaceCrossEncoder]:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        logger.info("Using CUDA device")

    split_docs = load_split_documents(splits_file)
    dense_retriever = create_dense_retriever(chroma_dir, collection_name, k=dense_k)
    sparse_retriever = create_bm25_retriever(split_docs, k=sparse_k)
    hybrid_retriever = create_hybrid_retriever(dense_retriever, sparse_retriever)

    logger.info("Loading cross-encoder model %s", RERANKER_MODEL)
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)

    return hybrid_retriever, cross_encoder


def query_naked(
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    model_name: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """Query the LLM directly without RAG context."""
    logger.info("Processing naked query: %s", question[:50])

    with langfuse.start_as_current_observation(
        as_type="span", name="naked_query"
    ) as query_span:
        with propagate_attributes(session_id=session_id, user_id=user_id):
            query_span.update_trace(
                input={"question": question},
                metadata={
                    "model": model_name,
                    "temperature": temperature,
                    "mode": "naked",
                },
            )

            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                keep_alive=-1,
            )
            prompt = ChatPromptTemplate.from_template(
                "You are a cybersecurity expert.\n\nQuestion: {question}"
            )
            chain = prompt | llm | StrOutputParser()

            langfuse_handler = CallbackHandler()
            answer = chain.invoke(
                {"question": question},
                config={"callbacks": [langfuse_handler]},
            )

            query_span.update_trace(output={"answer": answer})

    return answer


def query_with_sources(
    retriever: EnsembleRetriever,
    cross_encoder: HuggingFaceCrossEncoder,
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    top_n: int = RERANK_TOP_N,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    model_name: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
) -> tuple[str, list[dict[str, str]]]:
    logger.info("Processing query: %s", question[:50])

    with langfuse.start_as_current_observation(
        as_type="span", name="rag_query"
    ) as query_span:
        with propagate_attributes(session_id=session_id, user_id=user_id):
            query_span.update_trace(
                input={"question": question},
                metadata={
                    "model": model_name,
                    "temperature": temperature,
                    "embedding_model": EMBEDDING_MODEL,
                    "reranker": RERANKER_MODEL,
                    "top_n": top_n,
                    "relevance_threshold": relevance_threshold,
                },
            )

            with langfuse.start_as_current_observation(
                as_type="span", name="retrieval"
            ) as retrieval_span:
                raw_docs = retriever.invoke(question)
                retrieval_span.update(output={"count": len(raw_docs)})

            with langfuse.start_as_current_observation(
                as_type="span", name="rerank_filter"
            ) as rerank_span:
                docs, scores = rerank_and_filter(
                    question,
                    raw_docs,
                    cross_encoder,
                    top_n=top_n,
                    threshold=relevance_threshold,
                )
                rerank_span.update(
                    output={
                        "input_count": len(raw_docs),
                        "output_count": len(docs),
                        "scores": scores,
                    }
                )

            context = format_docs(docs) if docs else "No relevant context found."
            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                keep_alive=-1,
            )
            prompt = ChatPromptTemplate.from_template(ADAPTIVE_PROMPT)
            chain = prompt | llm | StrOutputParser()

            langfuse_handler = CallbackHandler()
            answer = chain.invoke(
                {"context": context, "question": question},
                config={"callbacks": [langfuse_handler]},
            )

            sources = [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("url", "Unknown"),
                    "language": doc.metadata.get("language", "Unknown"),
                    "urldate": doc.metadata.get("urldate", "Unknown"),
                }
                for doc in docs
            ]

            query_span.update_trace(output={"answer": answer, "sources": sources})

    return answer, sources
