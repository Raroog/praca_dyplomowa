def debug_retrieval(
    splits_file: Path,
    term: str,
    retriever=None,
    hybrid_retriever=None,
    reranker_model: str = RERANKER_MODEL,
) -> None:
    """Debug why a term isn't being retrieved."""
    docs = load_split_documents(splits_file)
    term_lower = term.lower()

    # 1. Check raw corpus
    print(f"\n{'=' * 80}")
    print(f"SEARCHING FOR: '{term}'")
    print("=" * 80)

    matches = []
    for i, doc in enumerate(docs):
        pos = doc.page_content.lower().find(term_lower)
        if pos != -1:
            snippet = doc.page_content[max(0, pos - 30) : pos + len(term) + 50]
            matches.append((i, doc.metadata.get("title", "?"), snippet))

    print(f"\n[1] RAW CORPUS: Found in {len(matches)} chunks")
    for i, title, snippet in matches[:5]:
        print(f"  Chunk {i}: {title}")
        print(f"    ...{snippet}...")

    if not matches:
        print("  Term NOT FOUND in corpus. Check your splits file or indexing.")
        return

    # 2. Test BM25 directly
    print(f"\n[2] BM25 DIRECT TEST:")
    bm25 = BM25Retriever.from_documents(
        docs,
        k=10,
        preprocess_func=lambda s: re.findall(r"\b\w+\b", s.lower()),
    )
    bm25_results = bm25.invoke(term)
    for i, doc in enumerate(bm25_results[:5]):
        has_term = term_lower in doc.page_content.lower()
        print(
            f"  {i + 1}. {doc.metadata.get('title', '?')} [contains term: {has_term}]"
        )

    # 3. Hybrid retriever (before reranking)
    if hybrid_retriever:
        print(f"\n[3] HYBRID (dense + bm25, no rerank):")
        hybrid_results = hybrid_retriever.invoke(term)
        for i, doc in enumerate(hybrid_results[:10]):
            has_term = term_lower in doc.page_content.lower()
            print(
                f"  {i + 1}. {doc.metadata.get('title', '?')} [contains term: {has_term}]"
            )

        # 4. After reranking
        print(f"\n[4] AFTER RERANKING ({reranker_model}):")
        cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
        pairs = [(term, doc.page_content) for doc in hybrid_results]
        scores = cross_encoder.score(pairs)

        ranked = sorted(zip(scores, hybrid_results), reverse=True, key=lambda x: x[0])
        for i, (score, doc) in enumerate(ranked[:10]):
            has_term = term_lower in doc.page_content.lower()
            title = doc.metadata.get("title", "?")[:60]
            print(f"  {i + 1}. [{score:.3f}] {title} [match: {has_term}]")

    # 5. Full retriever (if different from hybrid)
    if retriever and retriever != hybrid_retriever:
        print(f"\n[5] FULL RETRIEVER:")
        results = retriever.invoke(term)
        for i, doc in enumerate(results[:10]):
            has_term = term_lower in doc.page_content.lower()
            print(
                f"  {i + 1}. {doc.metadata.get('title', '?')} [contains term: {has_term}]"
            )


def test_rerankers(hybrid_retriever, query: str, docs_to_rerank: int = 20) -> None:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    candidates = hybrid_retriever.invoke(query)[:docs_to_rerank]
    term = query.lower()

    rerankers = [
        "cisco-ai/SecureBERT2.0-cross_encoder",
        "BAAI/bge-reranker-v2-m3",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]

    for model_name in rerankers:
        print(f"\n{model_name}:")
        try:
            encoder = HuggingFaceCrossEncoder(model_name=model_name)
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = encoder.score(pairs)

            ranked = sorted(zip(scores, candidates), reverse=True, key=lambda x: x[0])
            for i, (score, doc) in enumerate(ranked[:5]):
                has_term = term in doc.page_content.lower()
                title = doc.metadata.get("title", "?")[:50]
                print(f"  {i + 1}. [{score:.3f}] {title} [match: {has_term}]")
        except Exception as e:
            print(f"  Error: {e}")
