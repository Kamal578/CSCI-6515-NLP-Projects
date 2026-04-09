from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "project4" / "task2_reading_comprehension"
DEFAULT_REPORT_TEX = PROJECT_ROOT / "report" / "project4_report.tex"

TASK1_MODEL_FACTS = {
    "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
    "base_encoder": "bert-base-multilingual-uncased",
    "task_type": "5-class sentiment classification",
    "languages": ["English", "Dutch", "German", "French", "Spanish", "Italian"],
    "num_labels": 5,
    "max_positions": 512,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "case_sensitive": False,
}

TASK1_DEMO_REVIEWS = [
    {
        "label": "Positive review",
        "text": "Fantastic camera, fast shipping, and excellent battery life. I would buy it again.",
        "likely_label": "5 stars",
        "why": "Strong positive adjectives and a clear purchase recommendation usually push the classifier toward the top sentiment class.",
        "casing_note": "Capitalization is not needed for the signal here; the lexical content already carries strong positive polarity.",
    },
    {
        "label": "Mixed review",
        "text": "The screen is sharp, but the setup was confusing and the battery drains too quickly.",
        "likely_label": "2 to 3 stars",
        "why": "The sentence mixes praise with concrete complaints, so it resembles the kind of review that falls into a middle or mildly negative class.",
        "casing_note": "Because the model is uncased, it focuses more on words like 'confusing' and 'drains' than on surface formatting.",
    },
    {
        "label": "All-caps emphasis",
        "text": "THIS WAS AMAZING. The sound quality is great and the device feels premium.",
        "likely_label": "5 stars",
        "why": "The content is strongly positive, but the uncased model treats the all-caps emphasis mostly the same as lowercase text.",
        "casing_note": "This is a good live example for explaining what the model loses by removing case distinctions.",
    },
]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 4 Streamlit dashboard")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
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
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
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


def _safe_local_squad_lookup(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    lookup: dict[str, dict[str, str]] = {}
    for article in payload.get("data", []):
        title = str(article.get("title", ""))
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", ""))
            for qa in paragraph.get("qas", []):
                lookup[str(qa.get("id", ""))] = {
                    "title": title,
                    "question": str(qa.get("question", "")),
                    "context": context,
                }
    return lookup


def _fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a"


def _artifact_status_df(output_root: Path, report_tex: Path) -> pd.DataFrame:
    checks = [
        ("Task 1 report source", report_tex),
        ("Task 2 comparison", output_root / "comparison.csv"),
        ("Task 2 notes", output_root / "report_notes.md"),
        ("GloVe summary", output_root / "glove" / "summary.json"),
        ("GloVe history", output_root / "glove" / "history.csv"),
        ("GloVe predictions", output_root / "glove" / "predictions.json"),
        ("BERT summary", output_root / "bert" / "summary.json"),
        ("BERT history", output_root / "bert" / "history.csv"),
        ("BERT predictions", output_root / "bert" / "predictions.json"),
    ]
    rows: list[dict] = []
    for label, path in checks:
        rows.append({"artifact": label, "path": str(path), "status": "present" if path.exists() else "missing"})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_question_lookup(val_json: str | None, cache_dir: str | None) -> dict[str, dict[str, str]]:
    if val_json:
        return _safe_local_squad_lookup(Path(val_json).expanduser())

    try:
        from datasets import load_dataset
    except Exception:
        return {}

    try:
        dataset = load_dataset("squad", split="validation", cache_dir=cache_dir)
    except Exception:
        return {}

    lookup: dict[str, dict[str, str]] = {}
    for row in dataset:
        lookup[str(row.get("id", ""))] = {
            "title": str(row.get("title", "")),
            "question": str(row.get("question", "")),
            "context": str(row.get("context", "")),
        }
    return lookup


@st.cache_data(show_spinner=False)
def load_bundle(output_root: str, report_tex: str) -> dict:
    root = Path(output_root)
    report_path = Path(report_tex)
    glove_summary = _safe_json(root / "glove" / "summary.json")
    bert_summary = _safe_json(root / "bert" / "summary.json")
    config = bert_summary.get("config", {}) or glove_summary.get("config", {})
    return {
        "root": root,
        "report_tex_path": report_path,
        "glove_summary": glove_summary,
        "bert_summary": bert_summary,
        "glove_history": _safe_csv(root / "glove" / "history.csv"),
        "bert_history": _safe_csv(root / "bert" / "history.csv"),
        "comparison": _safe_csv(root / "comparison.csv"),
        "glove_predictions": _safe_json_list(root / "glove" / "predictions.json"),
        "bert_predictions": _safe_json_list(root / "bert" / "predictions.json"),
        "question_lookup": load_question_lookup(config.get("val_json"), config.get("cache_dir")),
        "report_notes": _safe_text(root / "report_notes.md"),
        "report_tex": _safe_text(report_path),
        "artifact_status": _artifact_status_df(root, report_path),
    }


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;700;800&display=swap');
html, body, [class*="css"] {
  font-family: "Manrope", sans-serif;
}
h1, h2, h3, h4 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.02em;
}
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2.2rem;
  max-width: 1280px;
}
.hero {
  background:
    radial-gradient(circle at top left, rgba(42,157,143,0.22), transparent 38%),
    radial-gradient(circle at top right, rgba(233,196,106,0.26), transparent 30%),
    linear-gradient(135deg, #f5fbfb 0%, #fcfaf4 100%);
  border: 1px solid #d8e4df;
  border-radius: 22px;
  padding: 1.35rem 1.5rem 1.15rem 1.5rem;
  margin-bottom: 1rem;
}
.hero-kicker {
  text-transform: uppercase;
  font-size: 0.78rem;
  letter-spacing: 0.12em;
  color: #2a6f67;
  font-weight: 800;
}
.hero-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: 2.1rem;
  line-height: 1.05;
  margin: 0.28rem 0 0.45rem 0;
  color: #17312f;
}
.hero-copy {
  color: #304745;
  font-size: 1rem;
  max-width: 62rem;
}
.flow-wrap {
  display: flex;
  gap: 0.75rem;
  align-items: stretch;
  margin-top: 0.95rem;
  margin-bottom: 0.4rem;
  flex-wrap: wrap;
}
.flow-card {
  min-width: 180px;
  flex: 1 1 220px;
  border: 1px solid #d7e2dc;
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(255,255,255,0.72);
}
.flow-card.accent {
  background: linear-gradient(135deg, rgba(42,157,143,0.12), rgba(233,196,106,0.16));
}
.flow-label {
  display: inline-block;
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #55736d;
  margin-bottom: 0.35rem;
  font-weight: 800;
}
.flow-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: 1rem;
  color: #183331;
  margin-bottom: 0.25rem;
}
.flow-copy {
  color: #3a5653;
  font-size: 0.93rem;
}
.mini-card {
  border: 1px solid #dbe4df;
  border-radius: 18px;
  padding: 0.95rem 1rem;
  background: linear-gradient(180deg, #ffffff 0%, #f9fbfa 100%);
  height: 100%;
}
.mini-card h4 {
  margin: 0 0 0.3rem 0;
  font-size: 1rem;
}
.mini-card p {
  margin: 0;
  color: #3e5653;
}
.badge {
  display: inline-block;
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
  font-size: 0.77rem;
  font-weight: 700;
  background: #eef6f5;
  color: #205b55;
  margin-right: 0.35rem;
  margin-bottom: 0.35rem;
}
.prediction-box {
  border: 1px solid #dbe4df;
  border-radius: 18px;
  padding: 1rem 1rem 0.9rem 1rem;
  background: #ffffff;
  height: 100%;
}
.prediction-box.winner {
  border-color: #2a9d8f;
  box-shadow: 0 0 0 1px rgba(42,157,143,0.12) inset;
  background: linear-gradient(180deg, #fbfffe 0%, #f3fbf8 100%);
}
.prediction-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #5a7772;
  margin-bottom: 0.4rem;
  font-weight: 800;
}
.prediction-text {
  font-size: 1.05rem;
  color: #183331;
  margin-bottom: 0.6rem;
}
.muted {
  color: #5d7774;
}
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #eff7f8 0%, #f8fbf6 100%);
  border: 1px solid #d7e4df;
  border-radius: 16px;
  padding: 8px 10px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(bundle: dict, output_root: Path) -> None:
    notes_preview = "Explore the Task 1 model analysis, Task 2 experiment comparison, and prediction-level differences."
    root_text = html.escape(str(output_root))
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-kicker">Project 4 Results UI</div>
  <div class="hero-title">BERT Sentiment Analysis + BiDAF Reading Comprehension</div>
  <div class="hero-copy">
    One dashboard for the theoretical BERT sentiment study and the implemented QA pipeline.
    Current artifact root: <code>{root_text}</code>.
    {notes_preview}
  </div>
  <div class="flow-wrap">
    <div class="flow-card">
      <div class="flow-label">Task 1</div>
      <div class="flow-title">Sentiment Model Anatomy</div>
      <div class="flow-copy">Model card facts, input/output contract, casing behavior, and Azerbaijani transfer analysis.</div>
    </div>
    <div class="flow-card accent">
      <div class="flow-label">Task 2</div>
      <div class="flow-title">GloVe vs Frozen BERT</div>
      <div class="flow-copy">BiDAF span prediction results, training history, and side-by-side prediction inspection.</div>
    </div>
    <div class="flow-card">
      <div class="flow-label">Artifacts</div>
      <div class="flow-title">Report + Outputs</div>
      <div class="flow-copy">Live file status, run commands, report notes, and raw output snapshots.</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _comparison_metrics(bundle: dict) -> tuple[float | None, float | None]:
    glove = bundle["glove_summary"].get("training", {})
    bert = bundle["bert_summary"].get("training", {})
    glove_f1 = glove.get("best_f1")
    bert_f1 = bert.get("best_f1")
    try:
        delta = float(bert_f1) - float(glove_f1)
    except Exception:
        delta = None
    return glove_f1 if glove_f1 is not None else None, delta


def _prediction_demo_examples(merged: pd.DataFrame) -> list[dict[str, str]]:
    if merged.empty:
        return []

    demos: list[dict[str, str]] = []

    bert_wins = merged[merged["winner"] == "bert"].sort_values("f1_delta_bert_minus_glove", ascending=False)
    if not bert_wins.empty:
        row = bert_wins.iloc[0]
        demos.append(
            {
                "label": "Strongest BERT win",
                "description": "Jump to the saved example where frozen BERT beats GloVe by the widest F1 margin.",
                "selected_id": str(row["id"]),
                "winner_filter": "bert",
                "text_filter": "",
            }
        )

    glove_wins = merged[merged["winner"] == "glove"].sort_values("f1_delta_bert_minus_glove", ascending=True)
    if not glove_wins.empty:
        row = glove_wins.iloc[0]
        demos.append(
            {
                "label": "Strongest GloVe win",
                "description": "Useful for showing that the baseline still wins on some individual questions.",
                "selected_id": str(row["id"]),
                "winner_filter": "glove",
                "text_filter": "",
            }
        )

    bert_exact = merged[merged.get("bert_exact_match", 0).fillna(0.0) >= 1.0]
    if not bert_exact.empty:
        row = bert_exact.sort_values("bert_score", ascending=False).iloc[0]
        demos.append(
            {
                "label": "Clean BERT exact match",
                "description": "Shows a saved row where the BERT variant hits the gold answer exactly.",
                "selected_id": str(row["id"]),
                "winner_filter": "all",
                "text_filter": "",
            }
        )

    tie_rows = merged[merged["winner"] == "tie"]
    if not tie_rows.empty:
        row = tie_rows.iloc[0]
        demos.append(
            {
                "label": "Tie example",
                "description": "A convenient example when both variants land on the same F1.",
                "selected_id": str(row["id"]),
                "winner_filter": "tie",
                "text_filter": "",
            }
        )

    return demos


def render_overview(bundle: dict, output_root: Path, report_tex: Path) -> None:
    st.subheader("Overview")
    artifact_df = bundle["artifact_status"]
    present_count = int((artifact_df["status"] == "present").sum()) if not artifact_df.empty else 0
    total_count = len(artifact_df)
    glove_summary = bundle["glove_summary"]
    bert_summary = bundle["bert_summary"]
    comparison = bundle["comparison"]
    glove_f1, delta_f1 = _comparison_metrics(bundle)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Artifacts ready", f"{present_count}/{total_count}")
    c2.metric("GloVe best F1", _fmt_float(glove_summary.get("training", {}).get("best_f1")))
    c3.metric("BERT best F1", _fmt_float(bert_summary.get("training", {}).get("best_f1")))
    c4.metric("BERT delta F1", _fmt_float(delta_f1) if delta_f1 is not None else "n/a")

    left, right = st.columns([1.3, 1])
    with left:
        if not comparison.empty:
            melted = comparison.melt(
                id_vars=["variant", "best_epoch"],
                value_vars=["exact_match", "f1"],
                var_name="metric",
                value_name="value",
            )
            fig = px.bar(
                melted,
                x="variant",
                y="value",
                color="metric",
                barmode="group",
                text_auto=".4f",
                title="Task 2 Metric Board",
                color_discrete_map={"exact_match": "#264653", "f1": "#2a9d8f"},
            )
            fig.update_layout(yaxis_title="score", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Task 2 comparison CSV found yet.")

    with right:
        checks = artifact_df.copy()
        if not checks.empty:
            fig = go.Figure(
                go.Pie(
                    labels=checks["status"],
                    values=checks.groupby("status").size().reindex(["present", "missing"], fill_value=0).tolist(),
                    hole=0.62,
                    marker=dict(colors=["#2a9d8f", "#e76f51"]),
                    sort=False,
                    textinfo="label+value",
                )
            )
            fig.update_layout(title="Artifact Coverage", margin=dict(l=10, r=10, t=50, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            """
<div class="mini-card">
  <h4>Task 1 Anchor</h4>
  <p>The report analyzes a public fine-tuned multilingual BERT sentiment model instead of training a fresh classifier from scratch.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
<div class="mini-card">
  <h4>Task 2 Engine</h4>
  <p>BiDAF is implemented in PyTorch, with two variants sharing the same span head: pretrained GloVe and frozen BERT word representations.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            f"""
<div class="mini-card">
  <h4>Current Run Context</h4>
  <p>Output root: <code>{output_root}</code><br/>Report source: <code>{report_tex}</code></p>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Project 4 Delivery Board**")
    delivery = pd.DataFrame(
        [
            {"item": "Task 1 report section", "status": "done" if report_tex.exists() else "missing"},
            {"item": "Task 2 QA pipeline", "status": "done" if glove_summary or bert_summary else "missing"},
            {"item": "Task 2 comparison artifacts", "status": "done" if not comparison.empty else "partial"},
            {"item": "Extra Task Streamlit UI", "status": "done"},
        ]
    )
    st.dataframe(delivery, use_container_width=True, hide_index=True)


def render_task1(bundle: dict) -> None:
    st.subheader("Task 1: BERT Sentiment Model Analysis")
    st.caption("This section is theoretical by design. The dashboard summarizes the report's model analysis.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", str(TASK1_MODEL_FACTS["num_labels"]))
    c2.metric("Max input length", str(TASK1_MODEL_FACTS["max_positions"]))
    c3.metric("Layers", str(TASK1_MODEL_FACTS["num_layers"]))
    c4.metric("Case sensitive", "No")

    st.markdown(
        f"""
<div>
  <span class="badge">{TASK1_MODEL_FACTS["model_name"]}</span>
  <span class="badge">{TASK1_MODEL_FACTS["task_type"]}</span>
  <span class="badge">multilingual base encoder</span>
  <span class="badge">uncased tokenizer</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Quick demo reviews**")
    st.caption("These are illustrative examples for presentation use. They demonstrate how the analyzed model would likely behave, not live inference from the dashboard.")
    task1_demo_label = st.selectbox(
        "Choose a quick sentiment demo",
        [demo["label"] for demo in TASK1_DEMO_REVIEWS],
        key="task1_demo_label",
    )
    selected_demo = next(demo for demo in TASK1_DEMO_REVIEWS if demo["label"] == task1_demo_label)
    demo_left, demo_right = st.columns([1.4, 1])
    with demo_left:
        st.code(selected_demo["text"], language="text")
    with demo_right:
        st.metric("Likely class", selected_demo["likely_label"])
        st.write(selected_demo["why"])
        st.caption(selected_demo["casing_note"])

    left, right = st.columns([1.15, 1])
    with left:
        io_df = pd.DataFrame(
            [
                {
                    "aspect": "Input",
                    "details": "Raw review text -> WordPiece tokenization -> input_ids + attention_mask (+ token_type_ids when used)",
                },
                {
                    "aspect": "Output",
                    "details": "Five logits mapped to 1..5 stars through softmax-based class selection",
                },
                {
                    "aspect": "Casing",
                    "details": "Lowercased before tokenization, so capitalization is not preserved as a signal",
                },
                {
                    "aspect": "Azerbaijani use",
                    "details": "Reasonable transfer baseline, but additional fine-tuning is needed for reliable deployment",
                },
            ]
        )
        st.markdown("**Model Contract**")
        st.dataframe(io_df, use_container_width=True, hide_index=True)

        transfer_df = pd.DataFrame(
            [
                {"factor": "Multilingual pretraining", "direction": "helps", "weight": 3},
                {"factor": "Subword tokenization", "direction": "helps", "weight": 2},
                {"factor": "No Azerbaijani fine-tuning", "direction": "hurts", "weight": -3},
                {"factor": "Agglutinative morphology mismatch", "direction": "hurts", "weight": -2},
                {"factor": "Review-domain alignment only", "direction": "hurts", "weight": -1},
            ]
        )
        fig = px.bar(
            transfer_df,
            x="weight",
            y="factor",
            orientation="h",
            color="direction",
            color_discrete_map={"helps": "#2a9d8f", "hurts": "#e76f51"},
            title="Azerbaijani Transfer Factors (heuristic, not measured)",
        )
        fig.update_layout(xaxis_title="relative effect", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Model Snapshot**")
        model_lines = [
            f"Base encoder: `{TASK1_MODEL_FACTS['base_encoder']}`",
            f"Languages in fine-tuning data: {', '.join(TASK1_MODEL_FACTS['languages'])}",
            f"Hidden size: {TASK1_MODEL_FACTS['hidden_size']}",
            f"Attention heads: {TASK1_MODEL_FACTS['num_heads']}",
            f"Sequence limit: {TASK1_MODEL_FACTS['max_positions']} subword positions",
        ]
        for line in model_lines:
            st.write(f"- {line}")

        st.markdown("**Casing Tradeoff**")
        st.code(
            "THIS WAS AMAZING\nthis was amazing\nThis Was Amazing",
            language="text",
        )
        st.write(
            "An uncased model tends to collapse these variants toward the same lexical form. "
            "That improves robustness on noisy reviews, but it also removes emphasis carried by capitalization."
        )

        st.markdown("**Report Source**")
        with st.expander("Preview `report/project4_report.tex`"):
            preview = bundle["report_tex"][:3500] if bundle["report_tex"] else "Report file not found."
            st.code(preview, language="latex")


def _variant_frame(bundle: dict, variant: str) -> pd.DataFrame:
    summary = bundle[f"{variant}_summary"]
    history = bundle[f"{variant}_history"].copy()
    if history.empty:
        return history
    history["variant"] = variant
    history["best_f1"] = summary.get("training", {}).get("best_f1")
    return history


def render_task2(bundle: dict) -> None:
    st.subheader("Task 2: Reading Comprehension Experiment Board")
    comparison = bundle["comparison"]
    glove_summary = bundle["glove_summary"]
    bert_summary = bundle["bert_summary"]

    if comparison.empty and not glove_summary and not bert_summary:
        st.warning("Task 2 outputs are missing. Run the Task 2 pipeline first.")
        return

    variant = st.radio("Focus variant", ["glove", "bert"], horizontal=True)
    summary = bundle[f"{variant}_summary"]
    history = bundle[f"{variant}_history"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best epoch", _fmt_int(summary.get("training", {}).get("best_epoch")))
    c2.metric("Best EM", _fmt_float(summary.get("training", {}).get("best_exact_match")))
    c3.metric("Best F1", _fmt_float(summary.get("training", {}).get("best_f1")))
    c4.metric("Train batches", _fmt_int(summary.get("dataset", {}).get("train_batches")))

    left, right = st.columns([1.2, 1])
    with left:
        all_hist = pd.concat(
            [_variant_frame(bundle, "glove"), _variant_frame(bundle, "bert")],
            ignore_index=True,
        )
        if not all_hist.empty:
            fig = px.line(
                all_hist,
                x="epoch",
                y="val_f1",
                color="variant",
                markers=True,
                title="Validation F1 by Epoch",
                color_discrete_map={"glove": "#264653", "bert": "#2a9d8f"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("History CSV files are not available.")

        if not comparison.empty:
            fig = px.scatter(
                comparison,
                x="exact_match",
                y="f1",
                size="val_examples",
                color="variant",
                text="variant",
                title="Variant Comparison: EM vs F1",
                color_discrete_map={"glove": "#264653", "bert": "#2a9d8f"},
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        dataset = summary.get("dataset", {})
        dataset_df = pd.DataFrame(
            [
                {"metric": "train_examples", "value": dataset.get("train_examples")},
                {"metric": "val_examples", "value": dataset.get("val_examples")},
                {"metric": "train_windows", "value": dataset.get("train_windows")},
                {"metric": "val_windows", "value": dataset.get("val_windows")},
            ]
        )
        fig = px.bar(
            dataset_df,
            x="metric",
            y="value",
            color="metric",
            title=f"{variant.upper()} data footprint",
            color_discrete_sequence=["#e9c46a", "#2a9d8f", "#264653", "#f4a261"],
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="count")
        st.plotly_chart(fig, use_container_width=True)

        config = summary.get("config", {})
        preset = "smoke" if config.get("smoke") else "medium" if config.get("medium") else "full/custom"
        st.markdown("**Run Profile**")
        st.write(f"- Variant: `{variant}`")
        st.write(f"- Preset: `{preset}`")
        st.write(f"- Device: `{config.get('device', 'n/a')}`")
        st.write(f"- Batch size: `{config.get('batch_size', 'n/a')}`")
        st.write(f"- Context window words: `{config.get('context_window_words', 'n/a')}`")

    st.markdown("**Comparison Table**")
    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    if bundle["report_notes"]:
        st.markdown("**Saved Task 2 Notes**")
        st.code(bundle["report_notes"], language="markdown")

    with st.expander(f"{variant.upper()} summary JSON"):
        st.json(summary)


def _build_prediction_frame(bundle: dict) -> pd.DataFrame:
    glove = pd.DataFrame(bundle["glove_predictions"])
    bert = pd.DataFrame(bundle["bert_predictions"])
    if glove.empty and bert.empty:
        return pd.DataFrame()

    rename_map = {
        "prediction_text": "prediction",
        "exact_match": "exact_match",
        "f1": "f1",
        "score": "score",
    }
    if not glove.empty:
        glove = glove.rename(columns={k: f"glove_{v}" for k, v in rename_map.items()})
    if not bert.empty:
        bert = bert.rename(columns={k: f"bert_{v}" for k, v in rename_map.items()})

    keep_glove = [c for c in glove.columns if c == "id" or c.startswith("glove_") or c == "gold_answers"]
    keep_bert = [c for c in bert.columns if c == "id" or c.startswith("bert_") or c == "gold_answers"]
    glove = glove[keep_glove] if not glove.empty else glove
    bert = bert[keep_bert] if not bert.empty else bert

    if glove.empty:
        merged = bert.copy()
    elif bert.empty:
        merged = glove.copy()
    else:
        merged = glove.merge(bert, on="id", how="outer", suffixes=("", "_bertdup"))
        if "gold_answers_bertdup" in merged.columns and "gold_answers" not in merged.columns:
            merged = merged.rename(columns={"gold_answers_bertdup": "gold_answers"})
        drop_cols = [c for c in merged.columns if c.endswith("_bertdup")]
        if drop_cols:
            merged = merged.drop(columns=drop_cols)

    if "gold_answers" in merged.columns:
        merged["gold_answers_text"] = merged["gold_answers"].apply(
            lambda xs: " | ".join(xs) if isinstance(xs, list) else str(xs)
        )
    else:
        merged["gold_answers_text"] = ""

    question_lookup = bundle.get("question_lookup", {})
    if question_lookup:
        meta_df = pd.DataFrame(
            [
                {
                    "id": question_id,
                    "title": payload.get("title", ""),
                    "question_text": payload.get("question", ""),
                    "context_text": payload.get("context", ""),
                }
                for question_id, payload in question_lookup.items()
            ]
        )
        if not meta_df.empty:
            merged = merged.merge(meta_df, on="id", how="left")
    for col in ("title", "question_text", "context_text"):
        if col not in merged.columns:
            merged[col] = ""

    required_cols = [
        "glove_prediction",
        "glove_exact_match",
        "glove_f1",
        "glove_score",
        "bert_prediction",
        "bert_exact_match",
        "bert_f1",
        "bert_score",
    ]
    for col in required_cols:
        if col not in merged.columns:
            merged[col] = None

    def winner(row: pd.Series) -> str:
        gf1 = row.get("glove_f1")
        bf1 = row.get("bert_f1")
        try:
            gf1 = float(gf1)
            bf1 = float(bf1)
        except Exception:
            return "unknown"
        if abs(gf1 - bf1) < 1e-12:
            return "tie"
        return "bert" if bf1 > gf1 else "glove"

    merged["winner"] = merged.apply(winner, axis=1)
    if "glove_f1" in merged.columns and "bert_f1" in merged.columns:
        merged["f1_delta_bert_minus_glove"] = merged["bert_f1"].fillna(0.0) - merged["glove_f1"].fillna(0.0)
    else:
        merged["f1_delta_bert_minus_glove"] = 0.0
    return merged.sort_values(by=["winner", "id"]).reset_index(drop=True)


def _prediction_card(label: str, text: str, f1: object, em: object, score: object, winner: bool) -> str:
    klass = "prediction-box winner" if winner else "prediction-box"
    pred = html.escape(text) if text else "<span class='muted'>No prediction recorded.</span>"
    return f"""
<div class="{klass}">
  <div class="prediction-label">{label}</div>
  <div class="prediction-text">{pred}</div>
  <div class="muted">F1: <strong>{_fmt_float(f1)}</strong> &nbsp; EM: <strong>{_fmt_float(em)}</strong> &nbsp; Score: <strong>{_fmt_float(score)}</strong></div>
</div>
    """


def render_predictions(bundle: dict) -> None:
    st.subheader("Prediction Inspector")
    merged = _build_prediction_frame(bundle)
    if merged.empty:
        st.info("Prediction files are not available yet.")
        return

    st.session_state.setdefault("prediction_winner_filter", "all")
    st.session_state.setdefault("prediction_text_filter", "")
    st.session_state.setdefault("prediction_row_index", 0)
    st.session_state.setdefault("prediction_selected_id", "")

    demos = _prediction_demo_examples(merged)
    if demos:
        st.markdown("**Quick demo jumps**")
        st.caption("One click jumps to useful saved examples for live presentation.")
        demo_cols = st.columns(len(demos))
        for idx, demo in enumerate(demos):
            if demo_cols[idx].button(demo["label"], key=f"prediction_demo_{idx}", use_container_width=True):
                st.session_state["prediction_winner_filter"] = demo["winner_filter"]
                st.session_state["prediction_text_filter"] = demo["text_filter"]
                st.session_state["prediction_selected_id"] = demo["selected_id"]
                st.session_state["prediction_row_index"] = 0
        selected_demo = next(
            (demo for demo in demos if demo["selected_id"] == st.session_state.get("prediction_selected_id")),
            None,
        )
        if selected_demo is not None:
            st.info(selected_demo["description"])

    long_rows: list[dict] = []
    for _, row in merged.iterrows():
        for variant in ("glove", "bert"):
            if f"{variant}_prediction" in row:
                long_rows.append(
                    {
                        "id": row["id"],
                        "variant": variant,
                        "prediction": row.get(f"{variant}_prediction"),
                        "f1": row.get(f"{variant}_f1"),
                        "exact_match": row.get(f"{variant}_exact_match"),
                        "score": row.get(f"{variant}_score"),
                    }
                )
    long_df = pd.DataFrame(long_rows)

    left, right = st.columns([1.1, 1])
    with left:
        winner_filter = st.selectbox(
            "Winner filter",
            ["all", "bert", "glove", "tie", "unknown"],
            key="prediction_winner_filter",
        )
        text_filter = st.text_input(
            "Search gold or prediction text",
            key="prediction_text_filter",
        ).strip().lower()
        view = merged.copy()
        if winner_filter != "all":
            view = view[view["winner"] == winner_filter]
        if text_filter:
            mask = (
                view["question_text"].astype(str).str.lower().str.contains(text_filter, na=False)
                | view["gold_answers_text"].astype(str).str.lower().str.contains(text_filter, na=False)
                | view["title"].astype(str).str.lower().str.contains(text_filter, na=False)
                | view["context_text"].astype(str).str.lower().str.contains(text_filter, na=False)
                | view.get("glove_prediction", pd.Series(index=view.index, dtype=str))
                .astype(str)
                .str.lower()
                .str.contains(text_filter, na=False)
                | view.get("bert_prediction", pd.Series(index=view.index, dtype=str))
                .astype(str)
                .str.lower()
                .str.contains(text_filter, na=False)
            )
            view = view[mask]
        view = view.reset_index(drop=True)

        st.dataframe(
            view[
                [
                    "id",
                    "winner",
                    "question_text",
                    "gold_answers_text",
                    "glove_prediction",
                    "bert_prediction",
                    "f1_delta_bert_minus_glove",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        if not long_df.empty:
            fig = px.scatter(
                long_df,
                x="score",
                y="f1",
                color="variant",
                hover_data=["id", "prediction"],
                title="Prediction Quality vs Model Score",
                color_discrete_map={"glove": "#264653", "bert": "#2a9d8f"},
            )
            st.plotly_chart(fig, use_container_width=True)

    if view.empty:
        st.warning("No rows match the current filters.")
        return

    target_id = st.session_state.get("prediction_selected_id", "")
    if target_id:
        matches = view.index[view["id"] == target_id].tolist()
        if matches:
            st.session_state["prediction_row_index"] = int(matches[0])
        st.session_state["prediction_selected_id"] = ""

    if len(view) == 1:
        selected_idx = 0
        st.session_state["prediction_row_index"] = 0
        st.caption("Only one row matches the current filters.")
    else:
        current_idx = int(st.session_state.get("prediction_row_index", 0))
        if current_idx < 0 or current_idx >= len(view):
            st.session_state["prediction_row_index"] = 0
        selected_idx = st.slider(
            "Inspect row",
            min_value=0,
            max_value=len(view) - 1,
            key="prediction_row_index",
        )
    row = view.iloc[selected_idx]

    if str(row.get("title", "")).strip():
        st.caption(f"Title: {row.get('title', '')}")

    st.markdown("**Question**")
    st.code(row.get("question_text", "") or "Question text not available.", language="text")

    st.markdown("**Gold Answers**")
    st.code(row.get("gold_answers_text", ""), language="text")

    with st.expander("Context passage"):
        st.write(row.get("context_text", "") or "Context text not available.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _prediction_card(
                "GloVe + BiDAF",
                str(row.get("glove_prediction", "")),
                row.get("glove_f1"),
                row.get("glove_exact_match"),
                row.get("glove_score"),
                row.get("winner") == "glove",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _prediction_card(
                "Frozen BERT + BiDAF",
                str(row.get("bert_prediction", "")),
                row.get("bert_f1"),
                row.get("bert_exact_match"),
                row.get("bert_score"),
                row.get("winner") == "bert",
            ),
            unsafe_allow_html=True,
        )


def render_artifacts(bundle: dict) -> None:
    st.subheader("Runbook and Artifact Explorer")
    st.markdown("**How to run the dashboard**")
    st.code(
        "bash scripts/run_project4_ui.sh\n"
        "streamlit run src/project4_dashboard.py -- --output-root outputs/project4/task2_reading_comprehension --report-tex report/project4_report.tex",
        language="bash",
    )

    st.markdown("**How to generate or refresh Task 2 outputs**")
    st.code(
        "SMOKE=1 bash scripts/run_project4_task2.sh\n"
        "MEDIUM=1 bash scripts/run_project4_task2.sh\n"
        "VARIANT=bert MEDIUM=1 bash scripts/run_project4_task2.sh",
        language="bash",
    )

    st.markdown("**Artifact Status**")
    st.dataframe(bundle["artifact_status"], use_container_width=True, hide_index=True)

    with st.expander("Task 2 comparison CSV"):
        comp = bundle["comparison"]
        if comp.empty:
            st.write("Comparison CSV not found.")
        else:
            st.dataframe(comp, use_container_width=True, hide_index=True)

    with st.expander("Task 2 report notes"):
        st.code(bundle["report_notes"] or "No notes file found.", language="markdown")

    with st.expander("Raw Task 1 report source"):
        st.code(bundle["report_tex"] or "Report source not found.", language="latex")


def main() -> None:
    args = parse_cli_args()
    output_root = Path(args.output_root).expanduser()
    report_tex = Path(args.report_tex).expanduser()

    st.set_page_config(
        page_title="Project 4 Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    if st.sidebar.button("Reload artifacts"):
        load_bundle.clear()

    bundle = load_bundle(str(output_root), str(report_tex))

    st.sidebar.title("Project 4 Dashboard")
    st.sidebar.caption("Sentiment theory + QA experiment explorer")
    st.sidebar.write(f"Output root: `{output_root}`")
    st.sidebar.write(f"Report source: `{report_tex}`")
    st.sidebar.write(
        f"Artifacts present: `{int((bundle['artifact_status']['status'] == 'present').sum())}/{len(bundle['artifact_status'])}`"
    )

    _render_hero(bundle, output_root)

    tabs = st.tabs(
        [
            "Overview",
            "Task 1",
            "Task 2",
            "Prediction Inspector",
            "Artifacts",
        ]
    )
    with tabs[0]:
        render_overview(bundle, output_root, report_tex)
    with tabs[1]:
        render_task1(bundle)
    with tabs[2]:
        render_task2(bundle)
    with tabs[3]:
        render_predictions(bundle)
    with tabs[4]:
        render_artifacts(bundle)


if __name__ == "__main__":
    main()
