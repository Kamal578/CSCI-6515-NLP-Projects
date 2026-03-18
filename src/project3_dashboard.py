from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from .project3_embeddings import EmbeddingSpace, load_text_vectors
except ImportError:
    from project3_embeddings import EmbeddingSpace, load_text_vectors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "project3"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3 interactive dashboard")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _resolve_artifact_path(path_str: str | None, output_root: Path) -> Path:
    if not path_str:
        return Path()
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    cands = [
        (PROJECT_ROOT / p).resolve(),
        (output_root / p).resolve(),
        p.resolve(),
    ]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


@st.cache_resource(show_spinner=False)
def _load_embedding_space_cached(vectors_path: str, max_words: int | None) -> EmbeddingSpace:
    return load_text_vectors(vectors_path, max_words=max_words)


def _load_embedding_space(vectors_path: Path, max_words: int | None) -> EmbeddingSpace:
    return _load_embedding_space_cached(str(vectors_path), max_words)


def _nearest_from_vector(
    space: EmbeddingSpace, query_vec: np.ndarray, top_k: int = 10, exclude: set[str] | None = None
) -> list[tuple[str, float]]:
    q = query_vec.astype(np.float32)
    q = q / np.clip(np.linalg.norm(q), 1e-12, None)
    sims = space.normed @ q
    if exclude:
        for word in exclude:
            idx = space.word_to_idx.get(word)
            if idx is not None:
                sims[idx] = -np.inf
    k = min(top_k, len(space.words) - len(exclude or set()))
    if k <= 0:
        return []
    idx = np.argpartition(-sims, kth=max(0, k - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(space.words[i], float(sims[i])) for i in idx]


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 2:
        return np.zeros((vectors.shape[0], 2), dtype=np.float32)
    x = vectors.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:2].T
    coords = x @ basis
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
    return coords[:, :2]


def _parse_words(raw: str) -> list[str]:
    if not raw.strip():
        return []
    words = [w.strip() for w in raw.replace("\n", ",").split(",")]
    words = [w for w in words if w]
    seen: set[str] = set()
    uniq: list[str] = []
    for w in words:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq


@st.cache_data(show_spinner=False)
def load_bundle(output_root: str) -> dict:
    root = Path(output_root)
    return {
        "task1_summary": _safe_json(root / "task1_matrices" / "summary.json"),
        "task2_summary": _safe_json(root / "task2_word2vec" / "summary.json"),
        "task3_summary": _safe_json(root / "task3_glove" / "summary.json"),
        "task4_summary": _safe_json(root / "task4_compare" / "summary.json"),
        "task5_summary": _safe_json(root / "task5_dl" / "summary.json"),
        "task6_summary": _safe_json(root / "task6_report" / "summary.json"),
        "task1_freq": _safe_csv(root / "task1_matrices" / "word_frequency_distribution.csv"),
        "task1_term_doc": _safe_csv(root / "task1_matrices" / "term_document_top_matrix.csv"),
        "task1_word_word": _safe_csv(root / "task1_matrices" / "word_word_top_matrix.csv"),
        "task2_neighbors": _safe_csv(root / "task2_word2vec" / "nearest_neighbors.csv"),
        "task2_analogy": _safe_csv(root / "task2_word2vec" / "analogy_results.csv"),
        "task3_neighbors": _safe_csv(root / "task3_glove" / "nearest_neighbors.csv"),
        "task3_analogy": _safe_csv(root / "task3_glove" / "analogy_results.csv"),
        "task4_comparison": _safe_csv(root / "task4_compare" / "comparison.csv"),
        "task5_leaderboard": _safe_csv(root / "task5_dl" / "leaderboard.csv"),
        "task5_metrics": _safe_csv(root / "task5_dl" / "metrics.csv"),
        "task5_confusion": _safe_json(root / "task5_dl" / "confusion_matrices.json"),
        "task5_reports_txt": _safe_text(root / "task5_dl" / "classification_reports.txt"),
        "task6_embed_table": _safe_csv(root / "task6_report" / "embedding_comparison_table.csv"),
        "task6_dl_table": _safe_csv(root / "task6_report" / "dl_results_table.csv"),
        "task6_md": _safe_text(root / "task6_report" / "report_artifacts.md"),
        "term_doc_heatmap_path": root / "task1_matrices" / "term_document_heatmap.png",
        "word_word_heatmap_path": root / "task1_matrices" / "word_word_heatmap.png",
    }


def _clean_word_word_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col0 = str(df.columns[0]).strip().lower()
    if col0.startswith("unnamed"):
        df = df.rename(columns={df.columns[0]: "term"}).set_index("term")
    return df


def _fmt_int(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a"


def _fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html, body, [class*="css"] {
  font-family: "Manrope", sans-serif;
}
h1, h2, h3 {
  font-family: "Space Grotesk", sans-serif;
}
.block-container {
  padding-top: 1.4rem;
  padding-bottom: 2rem;
}
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #eff7f8 0%, #f8fbf6 100%);
  border: 1px solid #d7e4df;
  border-radius: 12px;
  padding: 8px 10px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_overview(data: dict) -> None:
    st.subheader("Executive Overview")
    task1 = data["task1_summary"]
    task2 = data["task2_summary"]
    task3 = data["task3_summary"]
    task5 = data["task5_summary"]

    best = task5.get("best_overall", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Docs", _fmt_int(task1.get("dataset", {}).get("num_docs")))
    c2.metric("Vocabulary", _fmt_int(task1.get("dataset", {}).get("vocabulary_size")))
    c3.metric("Word2Vec hit@k", _fmt_float(task2.get("analogy_summary", {}).get("hit_at_k")))
    c4.metric("GloVe hit@k", _fmt_float(task3.get("analogy_summary", {}).get("hit_at_k")))
    c5.metric("Best Run", best.get("experiment_key", "n/a"))

    leaderboard = data["task5_leaderboard"]
    if not leaderboard.empty:
        fig = px.scatter(
            leaderboard,
            x="accuracy",
            y="f1_macro",
            color="architecture",
            symbol="feature_set",
            hover_data=["experiment_key"],
            title="Task 5 Run Landscape (Accuracy vs Macro-F1)",
        )
        st.plotly_chart(fig, use_container_width=True)

    notes = data["task4_summary"].get("qualitative_notes", [])
    if notes:
        st.markdown("**Task 4 Qualitative Notes**")
        for note in notes:
            st.write(f"- {note}")

    img_col1, img_col2 = st.columns(2)
    term_doc_img = data["term_doc_heatmap_path"]
    word_word_img = data["word_word_heatmap_path"]
    if term_doc_img.exists():
        img_col1.image(str(term_doc_img), caption="Task 1: Term-Document Heatmap")
    if word_word_img.exists():
        img_col2.image(str(word_word_img), caption="Task 1: Word-Word Heatmap")


def render_task1(data: dict) -> None:
    st.subheader("Task 1: Dataset and Matrix Analysis")
    summary = data["task1_summary"]
    dataset = summary.get("dataset", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens", _fmt_int(dataset.get("num_tokens")))
    c2.metric("Frequent Words (>=100)", _fmt_int(dataset.get("num_frequent_words")))
    c3.metric("Rare Words (<=2)", _fmt_int(dataset.get("num_rare_words")))

    freq_df = data["task1_freq"]
    if not freq_df.empty:
        st.markdown("**Frequency Explorer**")
        query = st.text_input("Filter term contains", value="", key="task1_term_filter")
        top_n = st.slider("Top-N terms", min_value=10, max_value=200, value=30, step=10)

        view = freq_df.copy()
        if query.strip():
            view = view[view["term"].astype(str).str.contains(query.strip(), case=False, na=False)]
        view = view.sort_values("count", ascending=False).head(top_n)

        st.dataframe(view, use_container_width=True)
        if not view.empty:
            fig = px.bar(view.iloc[::-1], x="count", y="term", orientation="h", title="Top Terms by Count")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Interactive Matrix View (Top Terms)**")
    matrix_choice = st.radio("Matrix", ["term-document", "word-word"], horizontal=True)
    if matrix_choice == "term-document":
        matrix_df = data["task1_term_doc"]
    else:
        matrix_df = _clean_word_word_df(data["task1_word_word"])

    if not matrix_df.empty:
        limit = st.slider("Matrix slice size", min_value=10, max_value=50, value=25, step=5)
        small = matrix_df.iloc[:limit, :limit]
        fig = px.imshow(
            small.values,
            x=[str(c) for c in small.columns],
            y=[str(i) for i in small.index],
            color_continuous_scale="Tealgrn",
            title=f"{matrix_choice} matrix ({limit}x{limit})",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Matrix CSV not found for Task 1.")


def render_embedding_panel(data: dict) -> None:
    st.subheader("Task 2/3: Word Embedding Explorer")
    model_choice = st.radio("Model", ["Word2Vec", "GloVe"], horizontal=True)

    if model_choice == "Word2Vec":
        summary = data["task2_summary"]
        neighbors = data["task2_neighbors"]
        analogy = data["task2_analogy"]
    else:
        summary = data["task3_summary"]
        neighbors = data["task3_neighbors"]
        analogy = data["task3_analogy"]

    model = summary.get("model", {})
    analogy_summary = summary.get("analogy_summary", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Vocab Size", _fmt_int(model.get("vocab_size")))
    c2.metric("Vector Dim", _fmt_int(model.get("vector_dim")))
    c3.metric("Analogy hit@k", _fmt_float(analogy_summary.get("hit_at_k")))

    st.markdown("**Nearest Neighbors**")
    if neighbors.empty:
        st.info("Neighbor file not found.")
    else:
        targets = sorted(neighbors["target"].dropna().unique().tolist())
        target = st.selectbox("Target word", options=targets, index=0 if targets else None)
        max_rank = int(neighbors["rank"].max()) if "rank" in neighbors.columns else 10
        top_k = st.slider("Top-K", min_value=3, max_value=max(3, max_rank), value=min(10, max(3, max_rank)))

        view = neighbors[(neighbors["target"] == target) & (neighbors["rank"] <= top_k)].copy()
        st.dataframe(view, use_container_width=True)
        if not view.empty:
            fig = px.line(
                view.sort_values("rank"),
                x="rank",
                y="cosine",
                markers=True,
                hover_data=["neighbor"],
                title=f"{model_choice}: cosine by rank for '{target}'",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Analogy Results**")
    if analogy.empty:
        st.info("Analogy file not found.")
    else:
        statuses = ["all"] + sorted(analogy["status"].dropna().unique().tolist())
        status_filter = st.selectbox("Status filter", options=statuses, index=0)
        view = analogy.copy()
        if status_filter != "all":
            view = view[view["status"] == status_filter]
        st.dataframe(view, use_container_width=True)


def render_task4(data: dict) -> None:
    st.subheader("Task 4: Word2Vec vs GloVe Comparison")
    summary = data["task4_summary"]
    df = data["task4_comparison"]

    if df.empty:
        st.info("Comparison table not found.")
    else:
        st.dataframe(df, use_container_width=True)
        metric = st.selectbox(
            "Metric chart",
            options=["analogy_hit_at_k", "coverage_ratio", "avg_neighbor_cosine_top3"],
            index=0,
        )
        fig = px.bar(df, x="model", y=metric, color="model", title=f"Model Comparison: {metric}")
        st.plotly_chart(fig, use_container_width=True)

    winner = summary.get("winner_by_metric", {})
    if winner:
        st.markdown("**Winner By Metric**")
        st.json(winner)

    notes = summary.get("qualitative_notes", [])
    if notes:
        st.markdown("**Qualitative Notes**")
        for note in notes:
            st.write(f"- {note}")


def render_task5(data: dict) -> None:
    st.subheader("Task 5: DL Experiment Grid")
    leaderboard = data["task5_leaderboard"]
    summary = data["task5_summary"]

    if leaderboard.empty:
        st.info("Leaderboard file not found.")
        return

    arch_choices = sorted(leaderboard["architecture"].dropna().unique().tolist())
    feat_choices = sorted(leaderboard["feature_set"].dropna().unique().tolist())

    selected_arch = st.multiselect("Architecture filter", arch_choices, default=arch_choices)
    selected_feat = st.multiselect("Feature filter", feat_choices, default=feat_choices)

    view = leaderboard[
        leaderboard["architecture"].isin(selected_arch) & leaderboard["feature_set"].isin(selected_feat)
    ].copy()
    view = view.sort_values("f1_macro", ascending=False)

    top_n = st.slider("Rows to display", min_value=5, max_value=max(5, len(view)), value=min(20, len(view)))
    st.dataframe(view.head(top_n), use_container_width=True)

    if not view.empty:
        fig = px.bar(
            view.head(top_n),
            x="experiment_key",
            y="f1_macro",
            color="architecture",
            hover_data=["feature_set", "accuracy"],
            title="Top Runs by Macro-F1",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

        pivot = (
            view[~view["architecture"].eq("logreg_baseline")]
            .pivot_table(index="architecture", columns="feature_set", values="f1_macro", aggfunc="max")
            .fillna(0.0)
        )
        if not pivot.empty:
            heat = px.imshow(
                pivot,
                text_auto=".3f",
                color_continuous_scale="YlGnBu",
                title="Macro-F1 Heatmap (architecture x feature)",
                aspect="auto",
            )
            st.plotly_chart(heat, use_container_width=True)

    confusion = data["task5_confusion"]
    labels = confusion.get("labels_order", [])
    matrices = confusion.get("matrices", {})
    if matrices:
        st.markdown("**Confusion Matrix Explorer**")
        exp_key = st.selectbox("Experiment key", options=sorted(matrices.keys()))
        cm_df = pd.DataFrame(matrices[exp_key], index=labels, columns=labels)
        normalize = st.checkbox("Normalize rows", value=False)
        if normalize:
            cm_df = cm_df.div(cm_df.sum(axis=1).replace(0, 1), axis=0)
        fig = px.imshow(
            cm_df,
            text_auto=".2f" if normalize else True,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title=f"Confusion Matrix: {exp_key}",
        )
        st.plotly_chart(fig, use_container_width=True)

    best = summary.get("best_overall", {})
    baseline_rows = summary.get("baseline", [])
    c1, c2 = st.columns(2)
    c1.markdown("**Best Overall**")
    c1.json(best)
    c2.markdown("**Baseline**")
    c2.json(baseline_rows[0] if baseline_rows else {})

    with st.expander("Classification reports text"):
        st.code(data["task5_reports_txt"] or "No report text found.", language="text")


def render_playground(data: dict, output_root: Path) -> None:
    st.subheader("Playground: Type and Explore")
    st.caption("Enter your own words and test the embedding spaces live.")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        model_choice = st.selectbox("Embedding model", ["Word2Vec", "GloVe"])
    with c2:
        load_profile = st.selectbox("Load profile", ["fast (40k words)", "balanced (80k words)", "full (all words)"])
    with c3:
        top_k = st.slider("Top-K", min_value=3, max_value=20, value=10)

    if model_choice == "Word2Vec":
        vectors_rel = data["task2_summary"].get("model", {}).get("vectors_path")
    else:
        vectors_rel = data["task3_summary"].get("model", {}).get("vectors_path")

    vectors_path = _resolve_artifact_path(vectors_rel, output_root)
    if not vectors_path.exists():
        st.error(f"Vectors file not found for {model_choice}: {vectors_path}")
        return

    max_words: int | None
    if load_profile == "fast (40k words)":
        max_words = 40000
    elif load_profile == "balanced (80k words)":
        max_words = 80000
    else:
        max_words = None

    with st.spinner(f"Loading {model_choice} vectors..."):
        space = _load_embedding_space(vectors_path, max_words=max_words)

    st.success(f"Loaded {model_choice}: vocab={space.vocab_size:,}, dim={space.dim}")

    st.markdown("### 1) Neighbor Search")
    query_word = st.text_input("Enter a word", value="azərbaycan", key="pg_neighbor_word").strip()
    if query_word:
        if not space.has_word(query_word):
            st.warning(f"'{query_word}' is OOV in loaded vocabulary.")
        else:
            rows = space.nearest_neighbors(query_word, top_k=top_k)
            df = pd.DataFrame(rows, columns=["neighbor", "cosine"])
            df.insert(0, "rank", np.arange(1, len(df) + 1))
            st.dataframe(df, use_container_width=True)
            if not df.empty:
                fig = px.bar(
                    df.iloc[::-1],
                    x="cosine",
                    y="neighbor",
                    orientation="h",
                    title=f"Neighbors for '{query_word}'",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2) Analogy Lab")
    a_col, b_col, c_col = st.columns(3)
    a = a_col.text_input("A", value="kişi", key="pg_a").strip()
    b = b_col.text_input("B", value="qadın", key="pg_b").strip()
    c = c_col.text_input("C", value="oğlan", key="pg_c").strip()
    if a and b and c:
        missing = [w for w in (a, b, c) if not space.has_word(w)]
        st.write(f"`{a}` : `{b}` :: `{c}` : `?`")
        if missing:
            st.warning(f"OOV words: {', '.join(missing)}")
        else:
            preds = space.analogy(a, b, c, top_k=top_k)
            pred_df = pd.DataFrame(preds, columns=["prediction", "cosine"])
            pred_df.insert(0, "rank", np.arange(1, len(pred_df) + 1))
            st.dataframe(pred_df, use_container_width=True)

    st.markdown("### 3) Word Similarity + Vector Blend")
    s1, s2 = st.columns(2)
    w1 = s1.text_input("Word 1", value="dövlət", key="pg_w1").strip()
    w2 = s2.text_input("Word 2", value="ölkə", key="pg_w2").strip()

    if w1 and w2 and space.has_word(w1) and space.has_word(w2):
        cos = float(space.normed[space.word_to_idx[w1]] @ space.normed[space.word_to_idx[w2]])
        st.metric("Cosine similarity", _fmt_float(cos))
    elif w1 and w2:
        missing = [w for w in (w1, w2) if not space.has_word(w)]
        if missing:
            st.warning(f"OOV words for similarity: {', '.join(missing)}")

    alpha = st.slider("Blend ratio (0 = Word 1, 1 = Word 2)", 0.0, 1.0, 0.5, 0.05)
    if w1 and w2 and space.has_word(w1) and space.has_word(w2):
        vec = (1.0 - alpha) * space.get_vector(w1) + alpha * space.get_vector(w2)
        blend = _nearest_from_vector(space, vec, top_k=top_k, exclude={w1, w2})
        blend_df = pd.DataFrame(blend, columns=["neighbor", "cosine"])
        blend_df.insert(0, "rank", np.arange(1, len(blend_df) + 1))
        st.dataframe(blend_df, use_container_width=True)

    st.markdown("### 4) Semantic Map (2D)")
    default_words = "azərbaycan, bakı, ankara, türkiyə, dövlət, tarix, elm, dil, film, mədəniyyət"
    raw_words = st.text_area("Words (comma or newline separated)", value=default_words, height=100)
    expand_anchor = st.text_input("Optional anchor word for auto-neighbors", value="", key="pg_anchor").strip()
    n_expand = st.slider("Neighbors to add from anchor", min_value=0, max_value=20, value=8, step=1)

    words = _parse_words(raw_words)
    if expand_anchor and space.has_word(expand_anchor) and n_expand > 0:
        extra = [w for w, _ in space.nearest_neighbors(expand_anchor, top_k=n_expand)]
        words = _parse_words(",".join(words + extra))

    in_vocab = [w for w in words if space.has_word(w)]
    missing = [w for w in words if not space.has_word(w)]

    if missing:
        st.caption(f"OOV ignored: {', '.join(missing[:20])}" + (" ..." if len(missing) > 20 else ""))

    if len(in_vocab) >= 2:
        mat = np.vstack([space.get_vector(w) for w in in_vocab])
        coords = _pca_2d(mat)
        plot_df = pd.DataFrame({"word": in_vocab, "x": coords[:, 0], "y": coords[:, 1]})
        fig = px.scatter(plot_df, x="x", y="y", text="word", title=f"{model_choice} semantic map")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plot_df.sort_values("word"), use_container_width=True)
    elif len(in_vocab) == 1:
        st.info("Need at least 2 in-vocabulary words to build a map.")
    else:
        st.info("No in-vocabulary words found from your input.")


def _download_button(path: Path, label: str, mime: str) -> None:
    if not path.exists():
        st.caption(f"Missing: {path}")
        return
    st.download_button(label=label, data=path.read_bytes(), file_name=path.name, mime=mime)


def render_task6(data: dict, output_root: Path) -> None:
    st.subheader("Task 6: Report Artifacts")
    st.markdown("**Summary**")
    st.json(data["task6_summary"])

    embed_table = data["task6_embed_table"]
    dl_table = data["task6_dl_table"]

    if not embed_table.empty:
        st.markdown("**Embedding Comparison Table**")
        st.dataframe(embed_table, use_container_width=True)

    if not dl_table.empty:
        st.markdown("**DL Results Table**")
        st.dataframe(dl_table, use_container_width=True)

    st.markdown("**Artifact Downloads**")
    c1, c2, c3 = st.columns(3)
    with c1:
        _download_button(output_root / "task6_report" / "embedding_comparison_table.csv", "Download embedding table", "text/csv")
    with c2:
        _download_button(output_root / "task6_report" / "dl_results_table.csv", "Download DL table", "text/csv")
    with c3:
        _download_button(output_root / "task6_report" / "report_artifacts.md", "Download report notes", "text/markdown")

    md_text = data["task6_md"]
    if md_text:
        st.markdown("**Report Notes Preview**")
        st.markdown(md_text)


def main() -> None:
    args = parse_cli_args()
    st.set_page_config(
        page_title="Project 3 Interactive Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.title("Project 3 Interactive Dashboard")
    st.caption("Word embeddings, model comparison, and deep-learning experiment explorer")

    st.sidebar.header("Data Source")
    output_root_input = st.sidebar.text_input("Output root path", value=str(args.output_root))
    output_root = Path(output_root_input).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    st.sidebar.caption(f"Resolved path: {output_root}")
    if not output_root.exists():
        st.error(f"Output root does not exist: {output_root}")
        st.stop()

    data = load_bundle(str(output_root))

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Playground",
            "Task 1 Matrices",
            "Task 2/3 Embeddings",
            "Task 4 Comparison",
            "Task 5 DL Grid",
            "Task 6 Report",
        ],
    )

    if page == "Overview":
        render_overview(data)
    elif page == "Playground":
        render_playground(data, output_root)
    elif page == "Task 1 Matrices":
        render_task1(data)
    elif page == "Task 2/3 Embeddings":
        render_embedding_panel(data)
    elif page == "Task 4 Comparison":
        render_task4(data)
    elif page == "Task 5 DL Grid":
        render_task5(data)
    else:
        render_task6(data, output_root)


if __name__ == "__main__":
    main()
