import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import gradio as gr
import uvicorn
from clean_rag import query_naked, query_with_sources, setup_rag
from fastapi import FastAPI
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langfuse import get_client
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

langfuse = get_client()


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    user_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, str]]


class AppState:
    retriever: EnsembleRetriever | None = None
    cross_encoder: HuggingFaceCrossEncoder | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG components...")
    state.retriever, state.cross_encoder = setup_rag()
    logger.info("RAG ready")
    yield
    langfuse.flush()
    logger.info("Shutdown complete")


app = FastAPI(title="Cybersecurity RAG", lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    answer, sources = query_with_sources(
        state.retriever,
        state.cross_encoder,
        request.question,
        session_id=request.session_id or str(uuid4()),
        user_id=request.user_id,
    )
    return QueryResponse(answer=answer, sources=sources)


def gradio_query(question: str, mode: str) -> tuple[str, str]:
    if not question.strip():
        return "", ""

    session_id = str(uuid4())

    if mode == "RAG":
        answer, sources = query_with_sources(
            state.retriever,
            state.cross_encoder,
            question,
            session_id=session_id,
        )
        sources_text = (
            "\n\n".join(f"**{s['title']}**\n{s['url']}" for s in sources)
            if sources
            else "No sources used."
        )
    else:
        answer = query_naked(question, session_id=session_id)
        sources_text = "*Naked mode — no retrieval*"

    return answer, sources_text


with gr.Blocks(title="Cybersecurity RAG") as demo:
    gr.Markdown("# 🔒 Cybersecurity RAG Assistant")

    question_input = gr.Textbox(
        label="Question",
        placeholder="Ask a cybersecurity question...",
        lines=2,
    )
    mode_select = gr.Radio(
        choices=["RAG", "Naked"],
        value="RAG",
        label="Mode",
    )
    submit_btn = gr.Button("Ask", variant="primary")
    answer_output = gr.Textbox(label="Answer", lines=12)
    sources_output = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=gradio_query,
        inputs=[question_input, mode_select],
        outputs=[answer_output, sources_output],
    )
    question_input.submit(
        fn=gradio_query,
        inputs=[question_input, mode_select],
        outputs=[answer_output, sources_output],
    )

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
