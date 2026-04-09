from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .project4_inference import load_qa_bundle, load_sentiment_bundle, predict_qa_answer, predict_sentiment
except ImportError:
    from src.project4_inference import load_qa_bundle, load_sentiment_bundle, predict_qa_answer, predict_sentiment


DEFAULT_SENTIMENT_ROOT = PROJECT_ROOT / "outputs" / "project4" / "task1_sentiment"
DEFAULT_QA_ROOT = PROJECT_ROOT / "outputs" / "project4" / "task2_reading_comprehension"
DEFAULT_REPORT_TEX = PROJECT_ROOT / "report" / "project4_report.tex"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 4 interactive dashboard")
    parser.add_argument("--sentiment-root", default=str(DEFAULT_SENTIMENT_ROOT))
    parser.add_argument("--qa-root", default=str(DEFAULT_QA_ROOT))
    parser.add_argument("--report-tex", default=str(DEFAULT_REPORT_TEX))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_json_list(path: Path) -> list[dict]:
    payload = _safe_json(path)
    return payload if isinstance(payload, list) else []


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


def _available_qa_variants(qa_root: Path) -> list[str]:
    variants: list[str] = []
    for variant in ("glove", "bert"):
        if (qa_root / variant / "summary.json").exists():
            variants.append(variant)
    return variants


@st.cache_resource(show_spinner=False)
def _cached_sentiment_bundle(model_root: str):
    return load_sentiment_bundle(model_root)


@st.cache_resource(show_spinner=False)
def _cached_qa_bundle(variant_root: str):
    return load_qa_bundle(variant_root)


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;700&display=swap');

:root {
  --ink: #163330;
  --muted: #567270;
  --line: #d2e1dc;
  --mint: #dff6ef;
  --sand: #f5efe1;
  --accent: #1f8f7d;
  --accent-dark: #0f5e53;
  --rose: #f3d7cb;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", sans-serif;
}

h1, h2, h3 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.02em;
  color: var(--ink);
}

.block-container {
  max-width: 1260px;
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

.hero {
  background:
    radial-gradient(circle at top left, rgba(31,143,125,0.18), transparent 34%),
    radial-gradient(circle at top right, rgba(214,146,88,0.16), transparent 32%),
    linear-gradient(135deg, #fbfdfa 0%, #f7f3e8 100%);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 1.25rem 1.35rem 1.05rem 1.35rem;
  margin-bottom: 1rem;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--accent-dark);
  font-size: 0.76rem;
  font-weight: 700;
}

.hero-title {
  font-size: 2.15rem;
  margin: 0.25rem 0 0.45rem 0;
}

.hero-copy {
  color: var(--muted);
  max-width: 68rem;
}

.panel {
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 1rem 1rem 0.9rem 1rem;
  background: linear-gradient(180deg, #ffffff 0%, #fbfcfb 100%);
}

.result-card {
  border: 1px solid var(--line);
  border-radius: 20px;
  background: white;
  padding: 0.95rem 1rem 0.8rem 1rem;
}

.result-card strong {
  color: var(--ink);
}

.answer {
  font-size: 1.12rem;
  color: var(--ink);
  margin-top: 0.35rem;
}

.fine {
  color: var(--muted);
  font-size: 0.93rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(sentiment_root: Path, qa_root: Path) -> None:
    st.markdown(
        f"""
<div class="hero">
  <div class="eyebrow">Project 4 Interactive UI</div>
  <div class="hero-title">Azerbaijani BERT Sentiment + BiDAF Reading Comprehension</div>
  <div class="hero-copy">
    Live inference for both tasks, backed by locally saved training artifacts. Sentiment uses a fine-tuned
    BERT classifier on Azerbaijani reviews. Reading comprehension reloads the BiDAF checkpoints and predicts
    spans from user-supplied context and questions.
    Current roots: <code>{sentiment_root}</code> and <code>{qa_root}</code>.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _artifact_rows(sentiment_root: Path, qa_root: Path, report_tex: Path) -> list[dict[str, str]]:
    checks = [
        ("Task 1 summary", sentiment_root / "summary.json"),
        ("Task 1 model", sentiment_root / "model"),
        ("Task 1 history", sentiment_root / "history.csv"),
        ("Task 2 comparison", qa_root / "comparison.csv"),
        ("Task 2 BERT summary", qa_root / "bert" / "summary.json"),
        ("Task 2 GloVe summary", qa_root / "glove" / "summary.json"),
        ("Report source", report_tex),
    ]
    return [
        {"artifact": name, "path": str(path), "status": "present" if path.exists() else "missing"}
        for name, path in checks
    ]


def render_overview(sentiment_root: Path, qa_root: Path, report_tex: Path) -> None:
    sentiment_summary = _safe_json(sentiment_root / "summary.json")
    comparison = _safe_csv(qa_root / "comparison.csv")
    artifact_df = pd.DataFrame(_artifact_rows(sentiment_root, qa_root, report_tex))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Artifacts ready", f"{int((artifact_df['status'] == 'present').sum())}/{len(artifact_df)}")
    c2.metric("Sentiment best val F1", f"{sentiment_summary.get('training', {}).get('best_validation_f1_macro', 0):.4f}" if sentiment_summary else "n/a")
    if not comparison.empty and "variant" in comparison:
        bert_row = comparison[comparison["variant"] == "bert"]
        glove_row = comparison[comparison["variant"] == "glove"]
        bert_f1 = float(bert_row["f1"].iloc[0]) if not bert_row.empty else None
        glove_f1 = float(glove_row["f1"].iloc[0]) if not glove_row.empty else None
        c3.metric("QA BERT F1", f"{bert_f1:.4f}" if bert_f1 is not None else "n/a")
        c4.metric("QA delta F1", f"{(bert_f1 - glove_f1):+.4f}" if bert_f1 is not None and glove_f1 is not None else "n/a")
    else:
        c3.metric("QA BERT F1", "n/a")
        c4.metric("QA delta F1", "n/a")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Delivery Snapshot")
        st.dataframe(artifact_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Run Commands")
        st.code("bash scripts/run_project4_task1.sh", language="bash")
        st.code("bash scripts/run_project4_task2.sh", language="bash")
        st.code("bash scripts/run_project4_ui.sh", language="bash")
        st.markdown('</div>', unsafe_allow_html=True)


def render_sentiment_tab(sentiment_root: Path) -> None:
    summary = _safe_json(sentiment_root / "summary.json")
    history = _safe_csv(sentiment_root / "history.csv")

    st.subheader("Task 1: Azerbaijani Sentiment Analysis")
    if not summary:
        st.warning("Task 1 artifacts are missing. Run `bash scripts/run_project4_task1.sh` first.")
        return

    model_dir = sentiment_root / "model"
    left, right = st.columns([1.2, 1])
    with left:
        default_text = "Bu tətbiq çox rahatdır, amma son yenilənmədən sonra bəzən gec açılır."
        text = st.text_area("Azerbaijani review", value=default_text, height=140)
        if st.button("Predict Sentiment", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading Task 1 model and running inference..."):
                    bundle = _cached_sentiment_bundle(str(sentiment_root))
                    prediction = predict_sentiment(bundle, text)
                probs = pd.DataFrame(
                    {
                        "label": list(prediction["probabilities"].keys()),
                        "probability": list(prediction["probabilities"].values()),
                    }
                )
                st.markdown(
                    f"""
<div class="result-card">
  <strong>Predicted label</strong>
  <div class="answer">{prediction["predicted_label"]}</div>
  <div class="fine">Confidence: {prediction["confidence"]:.4f}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
                fig = px.bar(
                    probs,
                    x="label",
                    y="probability",
                    color="probability",
                    color_continuous_scale=["#dff6ef", "#1f8f7d"],
                )
                fig.update_layout(height=320, coloraxis_showscale=False, xaxis_title="", yaxis_title="probability")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Sentiment inference failed: {exc}")
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Model Contract")
        model = summary.get("model", {})
        io_contract = summary.get("io_contract", {})
        st.write(f"Base model: `{model.get('model_name', 'n/a')}`")
        st.write(f"Labels: `{', '.join(model.get('labels', []))}`")
        st.write(f"Max length: `{model.get('max_length', 'n/a')}`")
        st.write(f"Case sensitive: `{'yes' if model.get('case_sensitive') else 'no'}`")
        st.write(f"Saved model dir: `{model_dir}`")
        strategy = io_contract.get("agglutinative_adaptation", {}).get("agglutinative_adaptation", [])
        if strategy:
            st.markdown("Agglutinative adaptation:")
            for line in strategy:
                st.write(f"- {line}")
        st.markdown('</div>', unsafe_allow_html=True)

    if not history.empty:
        st.markdown("### Training History")
        melted = history.melt(id_vars=["epoch"], value_vars=["val_f1_macro", "test_f1_macro"], var_name="metric", value_name="value")
        fig = px.line(melted, x="epoch", y="value", color="metric", markers=True, color_discrete_sequence=["#1f8f7d", "#c96f3d"])
        fig.update_layout(height=320, xaxis_title="epoch", yaxis_title="macro-F1")
        st.plotly_chart(fig, use_container_width=True)

    morphology = summary.get("morphology", {})
    suffix_rows = morphology.get("common_azerbaijani_suffix_hits", [])
    if suffix_rows:
        st.markdown("### Morphology Diagnostics")
        st.caption("These diagnostics document how the review corpus reflects Azerbaijani agglutination and how often the tokenizer splits words into multiple pieces.")
        st.write(
            f"Average wordpieces per word: `{morphology.get('avg_wordpieces_per_word', 0):.3f}` | "
            f"Multi-piece word rate: `{morphology.get('multi_piece_word_rate', 0):.3f}`"
        )
        st.dataframe(pd.DataFrame(suffix_rows), use_container_width=True, hide_index=True)


def render_qa_tab(qa_root: Path) -> None:
    comparison = _safe_csv(qa_root / "comparison.csv")
    variants = _available_qa_variants(qa_root)

    st.subheader("Task 2: Reading Comprehension")
    st.caption("The QA model was trained on SQuAD-style extractive QA. The live demo predicts an answer span from the supplied context.")
    if not variants:
        st.warning("No QA checkpoints were found. Run `bash scripts/run_project4_task2.sh` first.")
        return

    selected_variants = st.multiselect("Variants to run", options=variants, default=variants)
    default_context = (
        "Azerbaijan's capital city is Baku. It is located on the western shore of the Caspian Sea "
        "and is the country's largest city."
    )
    default_question = "What is the capital city of Azerbaijan?"
    context = st.text_area("Context", value=default_context, height=180)
    question = st.text_input("Question", value=default_question)

    if st.button("Predict Answer", type="primary", key="qa_predict", use_container_width=True):
        if not selected_variants:
            st.warning("Choose at least one QA variant.")
        else:
            cols = st.columns(len(selected_variants))
            for idx, variant in enumerate(selected_variants):
                with cols[idx]:
                    try:
                        with st.spinner(f"Running {variant}..."):
                            bundle = _cached_qa_bundle(str(qa_root / variant))
                            result = predict_qa_answer(bundle, context, question)
                        st.markdown(
                            f"""
<div class="result-card">
  <strong>{variant.upper()}</strong>
  <div class="answer">{result["answer"] or "No answer found"}</div>
  <div class="fine">Score: {result["score"]:.4f} | Windows: {result["num_windows"]}</div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.error(f"{variant} inference failed: {exc}")

    if not comparison.empty:
        st.markdown("### Saved Validation Metrics")
        fig = px.bar(
            comparison.melt(id_vars=["variant"], value_vars=["exact_match", "f1"], var_name="metric", value_name="value"),
            x="variant",
            y="value",
            color="metric",
            barmode="group",
            text_auto=".4f",
            color_discrete_map={"exact_match": "#173f5f", "f1": "#1f8f7d"},
        )
        fig.update_layout(height=320, xaxis_title="", yaxis_title="score")
        st.plotly_chart(fig, use_container_width=True)


def render_artifacts_tab(sentiment_root: Path, qa_root: Path, report_tex: Path) -> None:
    report_text = _safe_text(report_tex)
    sentiment_predictions = _safe_json_list(sentiment_root / "test_predictions.json")
    qa_notes = _safe_text(qa_root / "report_notes.md")

    st.subheader("Artifacts and Report")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Report Preview")
        if report_text:
            st.text_area("`report/project4_report.tex`", value=report_text[:9000], height=420)
        else:
            st.info("No report source found yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Task 2 Notes")
        st.code(qa_notes or "No Task 2 notes found.", language="markdown")
        st.markdown("### Task 1 Prediction Snapshot")
        if sentiment_predictions:
            preview = pd.DataFrame(sentiment_predictions[:12])[
                ["text", "gold_label", "predicted_label", "confidence"]
            ]
            st.dataframe(preview, use_container_width=True, hide_index=True)
        else:
            st.info("No saved Task 1 predictions found.")
        st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    args = parse_cli_args()
    sentiment_root = Path(args.sentiment_root).expanduser()
    qa_root = Path(args.qa_root).expanduser()
    report_tex = Path(args.report_tex).expanduser()

    st.set_page_config(
        page_title="Project 4 NLP Systems",
        page_icon="🧠",
        layout="wide",
    )
    _inject_styles()
    _render_hero(sentiment_root, qa_root)

    overview_tab, sentiment_tab, qa_tab, artifacts_tab = st.tabs(
        ["Overview", "Sentiment", "Reading Comprehension", "Artifacts"]
    )

    with overview_tab:
        render_overview(sentiment_root, qa_root, report_tex)
    with sentiment_tab:
        render_sentiment_tab(sentiment_root)
    with qa_tab:
        render_qa_tab(qa_root)
    with artifacts_tab:
        render_artifacts_tab(sentiment_root, qa_root, report_tex)


if __name__ == "__main__":
    main()
