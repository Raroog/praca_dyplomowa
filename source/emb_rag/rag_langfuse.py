import logging
import os
import pickle
from pathlib import Path
from uuid import uuid4

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
    RERANK_TOP_N,
    RERANKER_MODEL,
    SPARSE_K,
    SPLITS_FILE,
    TEMPERATURE,
)
from dotenv import dotenv_values
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
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()


config = dotenv_values(".env")

OPENAI_API_KEY = config.get("OPENAI_API_KEY")

os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

# Initialize Langfuse client (reads from environment variables)
langfuse = get_client()


# Retriever setup (unchanged)
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
    return BM25Retriever.from_documents(documents, k=k)


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


def create_reranked_retriever(
    base_retriever,
    reranker_model: str = RERANKER_MODEL,
    top_n: int = RERANK_TOP_N,
) -> ContextualCompressionRetriever:
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
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logger.info("Using CUDA device")

    split_docs = load_split_documents(splits_file)
    dense_retriever = create_dense_retriever(chroma_dir, collection_name)
    sparse_retriever = create_bm25_retriever(split_docs)
    hybrid_retriever = create_hybrid_retriever(dense_retriever, sparse_retriever)

    return create_reranked_retriever(hybrid_retriever)


# RAG chain
def format_docs(docs: list[Document]) -> str:
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
    logger.info("Creating RAG chain with model %s", model_name)

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        keep_alive=-1,
        reasoning=False,
    )

    system_prompt = """You are a cybersecurity expert. Answer using ONLY the provided context.

When citing information, include [Source: {{title}}, {{url}}].

If the context does not contain sufficient information, state: "The provided context does not contain enough information to answer this question completely."

Do not use information outside the provided context.

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


# Query functions
def query(chain, question: str, session_id: str = None) -> str:
    """Run a single query with Langfuse tracking."""
    logger.info("Processing query: %s", question[:50])

    # Initialize handler - automatically inherits current trace context
    langfuse_handler = CallbackHandler()

    # Set trace attributes via metadata
    metadata = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id

    return chain.invoke(
        question, config={"callbacks": [langfuse_handler], "metadata": metadata}
    )


def query_with_sources(
    retriever, chain, question: str, session_id: str = None, user_id: str = None
) -> tuple[str, list[dict[str, str]]]:
    """Run a query and return answer with sources, tracked in Langfuse."""
    logger.info("Processing query with sources: %s", question[:50])

    # Create trace context
    with langfuse.start_as_current_observation(
        as_type="span", name="rag_query_with_sources"
    ) as query_span:
        # Set trace attributes
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
        ):
            query_span.update_trace(
                input={"question": question},
                metadata={
                    "model": OLLAMA_MODEL,
                    "temperature": TEMPERATURE,
                    "embedding_model": EMBEDDING_MODEL,
                    "reranker": RERANKER_MODEL,
                    "dense_k": DENSE_K,
                    "sparse_k": SPARSE_K,
                    "rerank_top_n": RERANK_TOP_N,
                },
            )

            # Retrieval span
            with langfuse.start_as_current_observation(
                as_type="span", name="retrieval"
            ) as retrieval_span:
                docs = retriever.invoke(question)
                retrieval_span.update(
                    output={
                        "num_docs": len(docs),
                        "doc_lengths": [len(d.page_content) for d in docs],
                    }
                )

            # Log retrieved chunks for debugging
            print("\n" + "=" * 80)
            print("RETRIEVED CHUNKS:")
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Chunk {i} ({len(doc.page_content)} chars) ---")
                print(doc.page_content[:200] + "...")
            print("=" * 80 + "\n")

            # Generation span with LangChain handler
            langfuse_handler = CallbackHandler()
            answer = chain.invoke(question, config={"callbacks": [langfuse_handler]})

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


# Initialization
def initialize_rag(
    chroma_dir: Path = CHROMA_DIR,
    splits_file: Path = SPLITS_FILE,
    collection_name: str = COLLECTION_NAME,
    model_name: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
):
    retriever = setup_retriever(chroma_dir, splits_file, collection_name)
    chain = create_rag_chain(retriever, model_name, temperature)
    return retriever, chain


def compare_prompts(
    retriever,
    query: str,
    model_name: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
) -> None:
    """Compare different prompts on the same retrieved context."""
    context_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in context_docs[:5]])

    print(f"\n{'=' * 80}")
    print("RETRIEVED SOURCES:")
    print("=" * 80)
    for i, doc in enumerate(context_docs[:5], 1):
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("url", "N/A")
        language = doc.metadata.get("language", "Unknown")
        urldate = doc.metadata.get("urldate", "Unknown")
        print(f"{i}. {title}")
        print(f"   URL: {url}")
        print(f"   Language: {language}")
        print(f"   Date accessed: {urldate}")
    print()

    prompts = {
        "no_context": """You are a cybersecurity expert.
Question: {question}""",
        "permissive": """You are a cybersecurity expert. Use the provided context to inform your answer,
but you may also draw on your general knowledge when context is incomplete.
Context:
{context}
Question: {question}""",
        "strict": """You are a cybersecurity expert. Answer ONLY using the provided context.
Do not use external knowledge. If context is insufficient, say so explicitly.
Context:
{context}
Question: {question}""",
        "guided": """You are a cybersecurity expert. Answer using the provided context.
Instructions:
1. Base your answer primarily on the context
2. Quote relevant parts: [Source: title, url]
3. If context doesn't fully answer, state: "Based on available context: [answer], but this doesn't cover [missing aspect]"
Context:
{context}
Question: {question}""",
    }

    llm = ChatOllama(model=model_name, temperature=temperature)

    for name, template in prompts.items():
        print(f"\n{'=' * 80}")
        print(f"PROMPT TYPE: {name}")
        print("=" * 80)

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"context": context, "question": query})
        print(response)
        print()


# Example usage
if __name__ == "__main__":
    retriever, rag_chain = initialize_rag()

    # Create a session for related queries
    session_id = str(uuid4())

    # test_query = input("Cybersecurity (malware) question: \n\n")

    # print(f"\n{'=' * 80}")
    # print(f"Query: {test_query}")
    # print(f"{'=' * 80}\n")

    # response, sources = query_with_sources(
    #     retriever,
    #     rag_chain,
    #     test_query,
    #     session_id=session_id,
    #     user_id="bartek",  # Optional
    # )

    # print(f"Response: {response}")
    # print(f"{'=' * 80}\n")
    # print(f"Sources: {sources}")

    test_query_compare = input("Cybersecurity (malware) question: \n\n")
    compare_prompts(retriever, test_query_compare)

    # Flush to ensure all data is sent
    langfuse.flush()
