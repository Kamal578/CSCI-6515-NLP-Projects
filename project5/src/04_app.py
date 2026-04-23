from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from common import (
    chroma_persistent_client,
    resolve_from_root,
    runtime_config_from_env,
)
from rag_prompts import rag_refine_template_str, rag_text_qa_template_str


DEMO_QUESTIONS = [
    {
        "title": "Scope 1/2 Emissions Targets",
        "question": "Which companies in the dataset discuss Scope 1 and Scope 2 emissions reduction targets, and how do they frame those targets?",
    },
    {
        "title": "Board-Level ESG Oversight",
        "question": "How do companies describe board oversight and governance accountability for ESG or sustainability strategy?",
    },
    {
        "title": "Workforce Diversity Metrics",
        "question": "What workforce diversity and inclusion metrics are reported most frequently across the ESG reports?",
    },
    {
        "title": "Renewable Energy Initiatives",
        "question": "What are the most commonly mentioned renewable energy initiatives, and which companies mention them?",
    },
    {
        "title": "Community Investment Themes",
        "question": "Which social impact themes appear most often in community investment sections of these ESG reports?",
    },
    {
        "title": "Supply Chain + Human Rights",
        "question": "What actions do companies report for responsible sourcing, supplier oversight, and human rights protections?",
    },
    {
        "title": "Net-Zero or Decarbonization",
        "question": "Which decarbonization pathways are discussed most often (efficiency, renewable procurement, electrification, offsets)?",
    },
    {
        "title": "Assurance and Reporting Frameworks",
        "question": "Which external reporting frameworks or assurance standards are most frequently referenced (e.g., GRI, SASB, TCFD)?",
    },
]


def collection_exists(client: chromadb.PersistentClient, target: str) -> bool:
    collections = client.list_collections()
    for item in collections:
        name = item.name if hasattr(item, "name") else str(item)
        if name == target:
            return True
    return False


def count_vectors_from_sqlite(chroma_path: Path, collection_name: str) -> int | None:
    db_path = chroma_path / "chroma.sqlite3"
    if not db_path.exists():
        return None
    query = """
    SELECT COUNT(*)
    FROM embeddings e
    JOIN segments s ON e.segment_id = s.id
    JOIN collections c ON s.collection = c.id
    WHERE c.name = ?
    """
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(query, (collection_name,)).fetchone()
        if row is None:
            return 0
        return int(row[0])
    except sqlite3.Error:
        return None


def collection_has_records(collection) -> bool:
    sample = collection.peek(limit=1)
    ids = sample.get("ids", []) if isinstance(sample, dict) else []
    return bool(ids)


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

        .stApp {
            background:
                radial-gradient(1000px 540px at 100% -5%, rgba(255, 182, 97, 0.24), transparent 56%),
                radial-gradient(800px 460px at -5% 10%, rgba(106, 137, 204, 0.22), transparent 58%),
                linear-gradient(180deg, #f4efe6 0%, #ece6dc 100%);
        }

        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'IBM Plex Serif', serif !important;
            letter-spacing: -0.01em;
        }

        .hero-wrap {
            border: 1px solid rgba(28, 44, 72, 0.16);
            border-radius: 18px;
            padding: 1.1rem 1.15rem;
            margin-bottom: 0.9rem;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(248, 243, 235, 0.90));
            box-shadow: 0 9px 30px rgba(24, 34, 45, 0.08);
        }

        .hero-eyebrow {
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #4c678a;
            font-weight: 700;
            margin: 0;
        }

        .hero-title {
            margin: 0.2rem 0 0;
            color: #1d2d44;
            font-size: 1.9rem;
            line-height: 1.15;
        }

        .hero-subtitle {
            margin: 0.35rem 0 0;
            color: #2f3d4d;
            font-size: 0.98rem;
        }

        .meta-pill {
            margin-top: 0.4rem;
            margin-right: 0.35rem;
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            color: #203a53;
            background: rgba(159, 183, 210, 0.32);
            border: 1px solid rgba(69, 100, 136, 0.25);
        }

        .answer-card {
            border-radius: 14px;
            padding: 0.95rem 0.95rem 0.5rem;
            border: 1px solid rgba(43, 63, 92, 0.15);
            background: rgba(255, 255, 255, 0.78);
        }

        .chunk-meta {
            color: #2f475f;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }

        .history-card {
            border-radius: 12px;
            border: 1px solid rgba(43, 63, 92, 0.18);
            background: rgba(255, 255, 255, 0.72);
            padding: 0.65rem 0.75rem;
            margin-bottom: 0.55rem;
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 11px !important;
            border: 1px solid rgba(54, 74, 102, 0.25) !important;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(244,238,228,0.94)) !important;
        }

        .stButton > button:hover {
            border-color: rgba(54, 74, 102, 0.50) !important;
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_session_state() -> None:
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "auto_submit" not in st.session_state:
        st.session_state["auto_submit"] = False
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "run_demo_set" not in st.session_state:
        st.session_state["run_demo_set"] = False
    if "compare_result" not in st.session_state:
        st.session_state["compare_result"] = None


def format_score(score: float | None) -> str:
    if score is None:
        return "N/A"
    return f"{score:.4f}"


def source_node_to_payload(source_node) -> dict[str, Any]:
    node = source_node.node
    metadata = node.metadata if node else {}
    return {
        "source_file": metadata.get("source_file", "unknown"),
        "ticker": metadata.get("ticker", "N/A"),
        "year": metadata.get("year", "N/A"),
        "chunk_id": metadata.get("chunk_id"),
        "token_count": metadata.get("token_count"),
        "similarity": source_node.score,
        "text": node.get_content(metadata_mode="none") if node else "",
    }


def response_to_history_entry(question: str, response, asked_at_utc: str | None = None) -> dict[str, Any]:
    timestamp = asked_at_utc or datetime.now(timezone.utc).isoformat()
    sources = [source_node_to_payload(s) for s in (response.source_nodes or [])]
    unique_files = sorted({s["source_file"] for s in sources if s.get("source_file")})
    return {
        "asked_at_utc": timestamp,
        "question": question,
        "answer": str(response),
        "source_count": len(sources),
        "source_files": unique_files,
        "sources": sources,
    }


def history_to_markdown(history: list[dict[str, Any]]) -> str:
    lines = ["# ESG RAG Session Export", ""]
    for idx, entry in enumerate(history, start=1):
        lines.append(f"## Query {idx}")
        lines.append(f"- Asked at (UTC): {entry.get('asked_at_utc', 'N/A')}")
        lines.append(f"- Question: {entry.get('question', '')}")
        lines.append("- Answer:")
        lines.append("")
        lines.append(str(entry.get("answer", "")))
        lines.append("")
        source_files = entry.get("source_files", [])
        lines.append(f"- Source files: {', '.join(source_files) if source_files else 'None'}")
        lines.append("")
    return "\n".join(lines)


def run_query_and_store(question: str, query_engine) -> dict[str, Any]:
    response = query_engine.query(question)
    entry = response_to_history_entry(question, response)
    st.session_state["query_history"].append(entry)
    return entry


def render_history_sidebar() -> None:
    history = st.session_state.get("query_history", [])
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Session History")
        st.caption(f"Stored queries: {len(history)}")

        if not history:
            st.caption("No queries asked yet.")
            return

        if st.button("Clear History", use_container_width=True):
            st.session_state["query_history"] = []
            st.rerun()

        latest = history[-1]
        st.download_button(
            "Export Latest (JSON)",
            data=json.dumps(latest, indent=2, ensure_ascii=True),
            file_name="latest_query.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Export Session (Markdown)",
            data=history_to_markdown(history),
            file_name="rag_session.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.markdown("#### Recent Queries")
        for idx, entry in enumerate(reversed(history[-5:]), start=1):
            question = entry.get("question", "")
            truncated = question if len(question) <= 82 else f"{question[:82]}..."
            st.markdown(
                f'<div class="history-card"><strong>{idx}.</strong> {truncated}<br/>'
                f"<small>Sources: {entry.get('source_count', 0)}</small></div>",
                unsafe_allow_html=True,
            )


def render_demo_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Demonstration Questions")
        st.caption("Load realistic ESG prompts to quickly showcase retrieval quality.")

        selected_idx = st.selectbox(
            "Question Bank",
            options=range(len(DEMO_QUESTIONS)),
            format_func=lambda idx: f"{idx + 1}. {DEMO_QUESTIONS[idx]['title']}",
        )
        selected_question = DEMO_QUESTIONS[selected_idx]["question"]
        st.caption(selected_question)

        col_a, col_b = st.columns(2)
        if col_a.button("Load", use_container_width=True):
            st.session_state["question_input"] = selected_question
            st.session_state["auto_submit"] = False
            st.rerun()
        if col_b.button("Load + Ask", use_container_width=True):
            st.session_state["question_input"] = selected_question
            st.session_state["auto_submit"] = True
            st.rerun()

        if st.button("Run Demo Set (3)", use_container_width=True):
            st.session_state["run_demo_set"] = True
            st.rerun()

        st.markdown("---")
        st.markdown("### Runtime")
        st.caption("Use the main panel to run RAG and inspect exact retrieved context chunks.")


@st.cache_resource(show_spinner=False)
def load_query_engine(
    chroma_path_value: str,
    collection_name: str,
    ollama_base_url: str,
    llm_model: str,
    embed_model_name: str,
    context_window: int,
    top_k: int,
):
    chroma_path = resolve_from_root(chroma_path_value)
    if not chroma_path.exists():
        raise RuntimeError(
            f"Vector DB path not found: {chroma_path}. Run `python src/02_build_index.py --rebuild` first."
        )

    client = chroma_persistent_client(chroma_path)
    if not collection_exists(client, collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' not found in {chroma_path}. "
            "Run `python src/02_build_index.py --rebuild` first."
        )

    collection = client.get_collection(collection_name)
    if not collection_has_records(collection):
        raise RuntimeError(
            f"Collection '{collection_name}' is empty. Run `python src/02_build_index.py --rebuild` first."
        )

    llm = Ollama(
        model=llm_model,
        base_url=ollama_base_url,
        request_timeout=180.0,
        temperature=0,
        context_window=context_window,
    )
    embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=ollama_base_url)

    Settings.llm = llm
    Settings.embed_model = embed_model

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    text_qa_template = PromptTemplate(rag_text_qa_template_str())
    refine_template = PromptTemplate(rag_refine_template_str())
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        llm=llm,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
    )
    vector_count = count_vectors_from_sqlite(chroma_path, collection_name)
    return query_engine, vector_count, llm


def render_response(entry: dict[str, Any]) -> None:
    st.markdown("### Answer")
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.write(entry.get("answer", ""))
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Retrieved Context Chunks", expanded=True):
        sources = entry.get("sources", [])
        if not sources:
            st.info("No source chunks were returned.")
            return

        for idx, source in enumerate(sources, start=1):
            st.markdown(f"#### Chunk {idx}")
            st.markdown(
                f'<div class="chunk-meta"><strong>Source:</strong> {source.get("source_file", "unknown")} | '
                f'<strong>Ticker:</strong> {source.get("ticker", "N/A")} | '
                f'<strong>Year:</strong> {source.get("year", "N/A")} | '
                f'<strong>Similarity:</strong> {format_score(source.get("similarity"))}</div>',
                unsafe_allow_html=True,
            )
            st.code(source.get("text", ""))


def run_demo_set(query_engine) -> None:
    st.markdown("### Demo Run")
    demo_subset = DEMO_QUESTIONS[:3]
    for item in demo_subset:
        question = item["question"]
        with st.spinner(f"Running: {item['title']}"):
            entry = run_query_and_store(question, query_engine)
        with st.container():
            st.markdown(f"**{item['title']}**")
            st.caption(question)
            st.write(entry["answer"])
            st.caption(
                "Sources: "
                + (", ".join(entry["source_files"]) if entry["source_files"] else "None")
            )


def render_compare_tab(llm, query_engine) -> None:
    st.markdown("### Baseline vs RAG Comparison")
    st.caption(
        "Ask the same question to the bare LLM (no retrieval) and to the full RAG pipeline. "
        "Use this to see where retrieval adds grounding and where the baseline hallucinates or drifts."
    )

    with st.form("compare_form"):
        compare_question = st.text_area(
            "Question",
            height=110,
            placeholder="Ask about emissions targets, governance, diversity metrics, or any ESG topic.",
        )
        run_col, clear_col = st.columns(2)
        compare_submitted = run_col.form_submit_button("Compare", use_container_width=True)
        compare_cleared = clear_col.form_submit_button("Clear", use_container_width=True)

    if compare_cleared:
        st.session_state["compare_result"] = None
        st.rerun()

    if compare_submitted:
        if not compare_question.strip():
            st.warning("Please enter a non-empty question.")
        else:
            q = compare_question.strip()
            with st.spinner("Running baseline (LLM only)..."):
                baseline_text = llm.complete(q).text.strip()
            with st.spinner("Running RAG (retrieval + generation)..."):
                rag_response = query_engine.query(q)
                rag_text = str(rag_response).strip()
            sources = [source_node_to_payload(s) for s in (rag_response.source_nodes or [])]
            st.session_state["compare_result"] = {
                "question": q,
                "baseline": baseline_text,
                "rag": rag_text,
                "sources": sources,
            }

    result = st.session_state.get("compare_result")
    if not result:
        st.caption("Results will appear here after you submit a question.")
        return

    st.markdown(f"**Question:** {result['question']}")
    st.markdown("---")

    col_b, col_r = st.columns(2, gap="large")

    with col_b:
        st.markdown("#### Baseline — LLM only")
        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )
        st.write(result["baseline"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("#### RAG — retrieval + generation")
        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )
        st.write(result["rag"])
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Retrieved Context Chunks", expanded=False):
        sources = result.get("sources", [])
        if not sources:
            st.info("No source chunks were returned.")
        else:
            for idx, source in enumerate(sources, start=1):
                st.markdown(f"#### Chunk {idx}")
                st.markdown(
                    f'<div class="chunk-meta"><strong>Source:</strong> {source.get("source_file", "unknown")} | '
                    f'<strong>Ticker:</strong> {source.get("ticker", "N/A")} | '
                    f'<strong>Year:</strong> {source.get("year", "N/A")} | '
                    f'<strong>Similarity:</strong> {format_score(source.get("similarity"))}</div>',
                    unsafe_allow_html=True,
                )
                st.code(source.get("text", ""))


def main() -> None:
    cfg = runtime_config_from_env()
    top_k = int(os.getenv("SIMILARITY_TOP_K", "3"))
    context_window = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "2048"))

    st.set_page_config(page_title="S&P 500 ESG RAG", page_icon="📊", layout="wide")
    inject_custom_styles()
    ensure_session_state()
    render_demo_sidebar()

    st.markdown(
        f"""
        <div class="hero-wrap">
            <p class="hero-eyebrow">LOCAL ESG RAG DEMONSTRATION</p>
            <h1 class="hero-title">S&amp;P 500 Sustainability Intelligence</h1>
            <p class="hero-subtitle">Query indexed ESG reports with transparent retrieval traces and source-grounded answers.</p>
            <span class="meta-pill">LLM: {cfg.llm_model}</span>
            <span class="meta-pill">Embedding: {cfg.embed_model}</span>
            <span class="meta-pill">top-k: {top_k}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        query_engine, vector_count, llm = load_query_engine(
            cfg.chroma_path,
            cfg.chroma_collection,
            cfg.ollama_base_url,
            cfg.llm_model,
            cfg.embed_model,
            context_window,
            top_k,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Vector Collection", cfg.chroma_collection)
    metric_b.metric("Indexed Chunks", "unavailable" if vector_count is None else f"{vector_count:,}")
    metric_c.metric("Context Window", context_window)

    render_history_sidebar()

    tab_ask, tab_compare = st.tabs(["Ask the System", "Baseline vs RAG"])

    with tab_ask:
        if st.session_state.pop("run_demo_set", False):
            run_demo_set(query_engine)
            st.markdown("---")

        st.markdown("### Ask the System")
        with st.form("query_form"):
            question = st.text_area(
                "Question",
                key="question_input",
                height=115,
                placeholder="Ask about emissions, governance, social impact, diversity, or reporting frameworks.",
            )
            submit_col, clear_col = st.columns(2)
            submitted = submit_col.form_submit_button("Generate Answer", use_container_width=True)
            clear_clicked = clear_col.form_submit_button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state["question_input"] = ""
            st.session_state["auto_submit"] = False
            st.rerun()

        auto_submit = st.session_state.pop("auto_submit", False)
        should_run = submitted or auto_submit

        if not should_run:
            st.caption("Tip: use the sidebar prompts or run the demo set to showcase retrieval quality.")
        elif not question.strip():
            st.warning("Please enter a non-empty question.")
        else:
            with st.spinner("Running retrieval and generation..."):
                entry = run_query_and_store(question.strip(), query_engine)
            render_response(entry)

    with tab_compare:
        render_compare_tab(llm, query_engine)


if __name__ == "__main__":
    main()
