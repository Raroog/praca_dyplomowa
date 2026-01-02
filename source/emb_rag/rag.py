import logging
import pickle
from pathlib import Path

import torch
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DENSE_K,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    RERANK_TOP_N,
    RERANKER_MODEL,
    SPARSE_K,
    SPLITS_FILE,
    TEMPERATURE,
)
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()
# torch.set_float32_matmul_precision("high")


# Retriever setup


def load_split_documents(splits_path: Path) -> list[Document]:
    """Load pre-split documents from pickle file for BM25 retriever."""
    logger.info("Loading split documents from %s", splits_path)
    with splits_path.open("rb") as f:
        return pickle.load(f)


def create_dense_retriever(
    persist_dir: Path,
    collection_name: str,
    model_name: str = EMBEDDING_MODEL,
    k: int = DENSE_K,
):
    """Load existing Chroma vectorstore and return as retriever."""
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
    """Create BM25 retriever from documents."""
    logger.info("Creating BM25 retriever from %d documents", len(documents))
    return BM25Retriever.from_documents(documents, k=k)


def create_hybrid_retriever(
    dense_retriever,
    sparse_retriever,
    weights: tuple[float, float] = (DENSE_WEIGHT, 1.0 - DENSE_WEIGHT),
) -> EnsembleRetriever:
    """Create ensemble retriever combining dense and sparse retrievers."""
    logger.info(
        "Creating hybrid retriever with weights: dense=%.2f, sparse=%.2f",
        weights[0],
        weights[1],
    )
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=list(weights),
    )


def create_reranked_retriever(
    base_retriever,
    reranker_model: str = RERANKER_MODEL,
    top_n: int = RERANK_TOP_N,
) -> ContextualCompressionRetriever:
    """Wrap retriever with cross-encoder reranker."""
    logger.info("Creating reranker with model %s, top_n=%d", reranker_model, top_n)

    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def setup_retriever(
    chroma_dir: Path = CHROMA_DIR,
    splits_file: Path = SPLITS_FILE,
    collection_name: str = COLLECTION_NAME,
) -> ContextualCompressionRetriever:
    """Set up the full hybrid retrieval pipeline."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logger.info("Using CUDA device")

    split_docs = load_split_documents(splits_file)
    dense_retriever = create_dense_retriever(chroma_dir, collection_name)
    sparse_retriever = create_bm25_retriever(split_docs)
    hybrid_retriever = create_hybrid_retriever(dense_retriever, sparse_retriever)

    return create_reranked_retriever(hybrid_retriever)


# =============================================================================
# RAG chain
# =============================================================================


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents for the prompt context."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("url", "N/A")
        formatted.append(f"[Source {i}]: {source}\nURL: {url}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(
    retriever,
    model_name: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
):
    """Create the RAG chain with retriever and Ollama LLM.

    Args:
        retriever: The retriever to use for fetching relevant documents.
        model_name: Ollama model name.
        temperature: Sampling temperature.
    """
    logger.info("Creating RAG chain with model %s", model_name)

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        keep_alive=-1,  # keep model loaded in memory
        reasoning=False,
    )

    system_prompt = """You are a cybersecurity expert. Use the provided context to inform and support your answer,
    but you may also draw on your general knowledge of malware techniques when the context is incomplete.

    When citing specific claims from context, reference the source url, give urldate, language and title.
    When adding information beyond the context, note it as general knowledge.

    Context:
    {context}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# =============================================================================
# Query functions
# =============================================================================


def query(chain, question: str) -> str:
    """Run a single query through the RAG chain."""
    logger.info("Processing query: %s", question[:50])
    return chain.invoke(question)


def query_with_sources(
    retriever, chain, question: str
) -> tuple[str, list[dict[str, str]]]:
    """Run a query and return both the answer and source metadata."""
    logger.info("Processing query with sources: %s", question[:50])
    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = [
        {
            "title": doc.metadata.get("title", "Unknown"),
            "url": doc.metadata.get("url", "Unknown"),
            "language": doc.metadata.get("language", "Unknown"),
            "urldate": doc.metadata.get("urldate", "Unknown"),
        }
        for doc in docs
    ]
    return answer, sources


# =============================================================================
# Initialization
# =============================================================================


def initialize_rag(
    chroma_dir: Path = CHROMA_DIR,
    splits_file: Path = SPLITS_FILE,
    collection_name: str = COLLECTION_NAME,
    model_name: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
):
    """Initialize and return the retriever and RAG chain.

    Returns:
        tuple: (retriever, rag_chain)
    """
    retriever = setup_retriever(chroma_dir, splits_file, collection_name)
    chain = create_rag_chain(retriever, model_name, temperature)
    return retriever, chain


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    retriever, rag_chain = initialize_rag()

    test_query = input("Cybersecurity (malware) question: \n\n")

    print(f"\n{'=' * 80}")
    print(f"Query: {test_query}")
    print(f"{'=' * 80}\n")

    # response = query(rag_chain, test_query)
    response, sources = query_with_sources(retriever, rag_chain, test_query)

    print(f"Response: {response}")
    print(f"{'=' * 80}\n")
    print(f"Sources: {sources}")
