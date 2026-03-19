from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .project2_task3_sentiment_data import load_sentiment_dataset
from .project2_task3_sentiment_preprocess import legacy_tokenize
from .project3_common import ensure_dir, write_json

PAD_IDX = 0
UNK_IDX = 1
ALL_FEATURES = ["count_svd", "tfidf_svd", "pmi_svd", "word2vec", "glove"]
ALL_ARCHS = ["rnn", "birnn", "lstm"]

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise ImportError("Task 5 requires PyTorch. Install torch first.") from exc


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextSequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int], max_len: int):
        self.sequences = [s[:max_len] for s in sequences]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.sequences[idx], self.labels[idx]


def collate_batch(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, labels = zip(*batch)
    lengths = torch.tensor([max(1, len(s)) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    x = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        else:
            x[i, 0] = UNK_IDX
    y = torch.tensor(labels, dtype=torch.long)
    return x, lengths, y


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        architecture: str,
        hidden_size: int,
        num_classes: int,
        dropout: float,
        trainable_embeddings: bool = False,
    ):
        super().__init__()
        if architecture not in ALL_ARCHS:
            raise ValueError(f"Unsupported architecture: {architecture}")

        emb_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor, freeze=not trainable_embeddings, padding_idx=PAD_IDX
        )

        in_dim = int(embedding_matrix.shape[1])
        self.architecture = architecture
        self.is_bidir = architecture == "birnn"

        if architecture == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False,
            )
        else:
            self.rnn = nn.RNN(
                input_size=in_dim,
                hidden_size=hidden_size,
                batch_first=True,
                nonlinearity="tanh",
                bidirectional=self.is_bidir,
            )

        out_dim = hidden_size * (2 if self.is_bidir else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out = self.rnn(packed)

        if self.architecture == "lstm":
            _, (h_n, _) = rnn_out
        else:
            _, h_n = rnn_out

        if self.is_bidir:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        return self.fc(self.dropout(last_hidden))


def build_sequence_model(
    architecture: str,
    embedding_matrix: np.ndarray,
    num_classes: int,
    hidden_size: int = 128,
    dropout: float = 0.2,
    trainable_embeddings: bool = False,
) -> SequenceClassifier:
    return SequenceClassifier(
        embedding_matrix=embedding_matrix,
        architecture=architecture,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout,
        trainable_embeddings=trainable_embeddings,
    )


def _parse_list(arg: str) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _tokenize_texts(texts: pd.Series) -> list[list[str]]:
    return [legacy_tokenize(t) for t in texts.fillna("").astype(str).tolist()]


def _build_vocab(tokenized: list[list[str]], min_freq: int, max_vocab: int) -> dict[str, int]:
    counts = Counter(tok for toks in tokenized for tok in toks)
    items = [(tok, c) for tok, c in counts.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:max_vocab]
    return {tok: i + 2 for i, (tok, _) in enumerate(items)}


def _encode(tokenized: list[list[str]], vocab: dict[str, int]) -> list[list[int]]:
    out: list[list[int]] = []
    for toks in tokenized:
        seq = [vocab.get(tok, UNK_IDX) for tok in toks]
        out.append(seq if seq else [UNK_IDX])
    return out


def _svd_with_padding(mat, target_dim: int, seed: int) -> np.ndarray:
    min_axis = min(mat.shape)
    if min_axis <= 2:
        return np.zeros((mat.shape[0], target_dim), dtype=np.float32)
    n_components = max(2, min(target_dim, min_axis - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    emb = svd.fit_transform(mat)
    if emb.shape[1] < target_dim:
        pad = np.zeros((emb.shape[0], target_dim - emb.shape[1]), dtype=emb.dtype)
        emb = np.hstack([emb, pad])
    return emb.astype(np.float32)


def _build_count_tfidf_vectors(
    train_texts: pd.Series,
    mode: str,
    embedding_dim: int,
    max_features: int,
    min_df: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    vec = CountVectorizer(
        lowercase=False,
        tokenizer=legacy_tokenize,
        token_pattern=None,
        min_df=min_df,
        max_features=max_features,
    )
    x_counts = vec.fit_transform(train_texts)
    x = TfidfTransformer().fit_transform(x_counts) if mode == "tfidf" else x_counts.astype(np.float64)
    term_doc = x.T
    emb = _svd_with_padding(term_doc, embedding_dim, seed)
    terms = vec.get_feature_names_out().tolist()
    token_to_vec = {t: emb[i] for i, t in enumerate(terms)}
    meta = {
        "mode": mode,
        "num_terms": len(terms),
        "input_shape": [int(x.shape[0]), int(x.shape[1])],
        "embedding_dim": embedding_dim,
    }
    return token_to_vec, meta


def _build_pmi_vectors(
    train_texts: pd.Series,
    embedding_dim: int,
    max_features: int,
    min_df: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    vec = CountVectorizer(
        lowercase=False,
        tokenizer=legacy_tokenize,
        token_pattern=None,
        min_df=min_df,
        max_features=max_features,
        binary=True,
    )
    x_bin = vec.fit_transform(train_texts).tocsr().astype(np.float64)
    if x_bin.nnz == 0:
        return {}, {"mode": "pmi", "num_terms": 0, "embedding_dim": embedding_dim}

    x_bin.data[:] = 1.0
    cooc = (x_bin.T @ x_bin).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    if cooc.nnz == 0:
        return {}, {"mode": "pmi", "num_terms": 0, "embedding_dim": embedding_dim}

    row_sum = np.asarray(cooc.sum(axis=1)).ravel()
    total = float(cooc.sum())
    rows, cols = cooc.nonzero()
    vals = cooc.data
    denom = np.clip(row_sum[rows] * row_sum[cols], 1e-12, None)
    pmi = np.log(np.clip((vals * total) / denom, 1e-12, None))
    ppmi = np.maximum(pmi, 0.0)
    ppmi_mat = sparse.csr_matrix((ppmi, (rows, cols)), shape=cooc.shape)
    emb = _svd_with_padding(ppmi_mat, embedding_dim, seed)

    terms = vec.get_feature_names_out().tolist()
    token_to_vec = {t: emb[i] for i, t in enumerate(terms)}
    meta = {
        "mode": "pmi",
        "num_terms": len(terms),
        "cooc_nnz": int(cooc.nnz),
        "embedding_dim": embedding_dim,
    }
    return token_to_vec, meta


def _load_subset_vectors(path: str, wanted_tokens: set[str], expected_dim: int) -> tuple[dict[str, np.ndarray], dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")

    token_to_vec: dict[str, np.ndarray] = {}
    dim: int | None = None
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()
        if len(first) == 2 and first[0].isdigit() and first[1].isdigit():
            dim = int(first[1])
        else:
            f.seek(0)

        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            tok = parts[0]
            if tok not in wanted_tokens:
                continue
            vals = parts[1:]
            if dim is None:
                dim = len(vals)
            if len(vals) != dim:
                continue
            try:
                vec = np.asarray([float(v) for v in vals], dtype=np.float32)
            except ValueError:
                continue

            if vec.shape[0] > expected_dim:
                vec = vec[:expected_dim]
            elif vec.shape[0] < expected_dim:
                pad = np.zeros(expected_dim - vec.shape[0], dtype=np.float32)
                vec = np.concatenate([vec, pad], axis=0)
            token_to_vec[tok] = vec
            if len(token_to_vec) == len(wanted_tokens):
                break

    meta = {
        "path": str(p),
        "requested_tokens": len(wanted_tokens),
        "loaded_tokens": len(token_to_vec),
        "coverage": (len(token_to_vec) / len(wanted_tokens)) if wanted_tokens else 0.0,
        "embedding_dim": expected_dim,
    }
    return token_to_vec, meta


def _embedding_matrix(vocab: dict[str, int], token_to_vec: dict[str, np.ndarray], dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 0.05, size=(len(vocab) + 2, dim)).astype(np.float32)
    mat[PAD_IDX] = 0.0
    for tok, idx in vocab.items():
        vec = token_to_vec.get(tok)
        if vec is not None:
            mat[idx] = vec.astype(np.float32)
    return mat


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, lengths, yb in loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            logits = model(xb, lengths)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(yb.numpy().tolist())
    return y_true, y_pred


@dataclass
class TrainResult:
    best_state: dict
    best_epoch: int
    best_val_macro_f1: float


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_macro = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

        y_true, y_pred = _predict(model, val_loader, device)
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if macro > best_val_macro:
            best_val_macro = macro
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return TrainResult(best_state=best_state, best_epoch=best_epoch, best_val_macro_f1=best_val_macro)


def _metrics(y_true: list[int], y_pred: list[int], labels_order: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels_order,
            digits=4,
            zero_division=0,
        ),
    }


def _prepare_feature_vectors(
    name: str,
    train_texts: pd.Series,
    vocab: dict[str, int],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    if name == "count_svd":
        return _build_count_tfidf_vectors(
            train_texts=train_texts,
            mode="count",
            embedding_dim=args.embedding_dim,
            max_features=args.svd_max_features,
            min_df=args.min_df,
            seed=args.seed,
        )
    if name == "tfidf_svd":
        return _build_count_tfidf_vectors(
            train_texts=train_texts,
            mode="tfidf",
            embedding_dim=args.embedding_dim,
            max_features=args.svd_max_features,
            min_df=args.min_df,
            seed=args.seed,
        )
    if name == "pmi_svd":
        return _build_pmi_vectors(
            train_texts=train_texts,
            embedding_dim=args.embedding_dim,
            max_features=args.pmi_max_features,
            min_df=args.min_df,
            seed=args.seed,
        )
    if name == "word2vec":
        return _load_subset_vectors(args.word2vec_vectors, set(vocab.keys()), args.embedding_dim)
    if name == "glove":
        return _load_subset_vectors(args.glove_vectors, set(vocab.keys()), args.embedding_dim)
    raise ValueError(f"Unsupported feature set: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3 Task 5: DL classification comparison.")
    p.add_argument("--train", default="data/external/train.csv")
    p.add_argument("--test", default="data/external/test.csv")
    p.add_argument("--text-col", default="content")
    p.add_argument("--score-col", default="score")
    p.add_argument("--label-mode", default="sentiment3", choices=["sentiment3", "score5", "binary"])
    p.add_argument("--out_dir", default="outputs/project3/task5_dl")

    p.add_argument("--features", default=",".join(ALL_FEATURES))
    p.add_argument("--architectures", default=",".join(ALL_ARCHS))
    p.add_argument("--word2vec_vectors", default="outputs/project3/task2_word2vec/vectors.txt")
    p.add_argument("--glove_vectors", default="outputs/project3/task3_glove/vectors.txt")

    p.add_argument("--embedding_dim", type=int, default=200)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_len", type=int, default=120)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--min_token_freq", type=int, default=2)
    p.add_argument("--max_vocab_tokens", type=int, default=50000)
    p.add_argument("--svd_max_features", type=int, default=25000)
    p.add_argument("--pmi_max_features", type=int, default=6000)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trainable_embeddings", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.max_train_samples = args.max_train_samples or 3000
        args.max_test_samples = args.max_test_samples or 800
        args.max_vocab_tokens = min(args.max_vocab_tokens, 8000)
        args.svd_max_features = min(args.svd_max_features, 4000)
        args.pmi_max_features = min(args.pmi_max_features, 2000)

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    features = _parse_list(args.features)
    archs = _parse_list(args.architectures)

    ds = load_sentiment_dataset(
        train_path=args.train,
        test_path=args.test,
        text_col=args.text_col,
        score_col=args.score_col,
        label_mode=args.label_mode,
    )
    train_df = ds.train_df.copy()
    test_df = ds.test_df.copy()
    labels_order = ds.labels_order

    if args.max_train_samples is not None:
        train_df = train_df.sample(
            n=min(args.max_train_samples, len(train_df)),
            random_state=args.seed,
        ).reset_index(drop=True)
    if args.max_test_samples is not None:
        test_df = test_df.sample(
            n=min(args.max_test_samples, len(test_df)),
            random_state=args.seed,
        ).reset_index(drop=True)

    train_part, val_part = train_test_split(
        train_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_df["label"],
    )
    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)

    label_to_id = {lbl: i for i, lbl in enumerate(labels_order)}
    y_train = [label_to_id[x] for x in train_part["label"].astype(str).tolist()]
    y_val = [label_to_id[x] for x in val_part["label"].astype(str).tolist()]
    y_test = [label_to_id[x] for x in test_df["label"].astype(str).tolist()]

    train_tok = _tokenize_texts(train_part[args.text_col])
    val_tok = _tokenize_texts(val_part[args.text_col])
    test_tok = _tokenize_texts(test_df[args.text_col])

    vocab = _build_vocab(train_tok, min_freq=args.min_token_freq, max_vocab=args.max_vocab_tokens)
    train_seq = _encode(train_tok, vocab)
    val_seq = _encode(val_tok, vocab)
    test_seq = _encode(test_tok, vocab)

    train_ds = TextSequenceDataset(train_seq, y_train, max_len=args.max_len)
    val_ds = TextSequenceDataset(val_seq, y_val, max_len=args.max_len)
    test_ds = TextSequenceDataset(test_seq, y_test, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    class_counts = np.bincount(y_train, minlength=len(labels_order)).astype(np.float32)
    class_weights = np.sum(class_counts) / np.clip(class_counts, 1.0, None)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    metrics_rows: list[dict] = []
    confusion_payload: dict[str, list[list[int]]] = {}
    report_lines: list[str] = []
    feature_meta: dict[str, dict] = {}

    for feat_name in features:
        token_to_vec, meta = _prepare_feature_vectors(feat_name, train_part[args.text_col], vocab, args)
        feature_meta[feat_name] = meta
        emb_matrix = _embedding_matrix(vocab, token_to_vec, args.embedding_dim, args.seed)

        for arch in archs:
            exp_key = f"{arch}__{feat_name}"
            model = build_sequence_model(
                architecture=arch,
                embedding_matrix=emb_matrix,
                num_classes=len(labels_order),
                hidden_size=args.hidden_size,
                dropout=args.dropout,
                trainable_embeddings=args.trainable_embeddings,
            ).to(device)

            tr = _train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights_t,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            model.load_state_dict(tr.best_state)

            y_true, y_pred = _predict(model, test_loader, device)
            m = _metrics(y_true, y_pred, labels_order)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_order))))

            metrics_rows.append(
                {
                    "experiment_key": exp_key,
                    "architecture": arch,
                    "feature_set": feat_name,
                    "accuracy": m["accuracy"],
                    "f1_macro": m["f1_macro"],
                    "f1_weighted": m["f1_weighted"],
                    "best_epoch": tr.best_epoch,
                    "best_val_macro_f1": tr.best_val_macro_f1,
                    "n_train": len(train_part),
                    "n_val": len(val_part),
                    "n_test": len(test_df),
                }
            )
            confusion_payload[exp_key] = cm.tolist()
            report_lines.append(f"## {exp_key}\n{m['classification_report']}")
            print(
                f"[{exp_key}] macro_f1={m['f1_macro']:.4f} "
                f"acc={m['accuracy']:.4f} val_best={tr.best_val_macro_f1:.4f}"
            )

    # Baseline: Logistic Regression + TF-IDF
    tfidf_vec = TfidfVectorizer(
        lowercase=False,
        tokenizer=legacy_tokenize,
        token_pattern=None,
        max_features=args.svd_max_features,
        min_df=args.min_df,
    )
    x_train_b = tfidf_vec.fit_transform(train_part[args.text_col])
    x_test_b = tfidf_vec.transform(test_df[args.text_col])
    baseline = LogisticRegression(max_iter=300, class_weight="balanced", random_state=args.seed)
    baseline.fit(x_train_b, y_train)
    y_pred_b = baseline.predict(x_test_b).tolist()
    m_b = _metrics(y_test, y_pred_b, labels_order)
    cm_b = confusion_matrix(y_test, y_pred_b, labels=list(range(len(labels_order))))
    base_key = "logreg_baseline__tfidf"
    metrics_rows.append(
        {
            "experiment_key": base_key,
            "architecture": "logreg_baseline",
            "feature_set": "tfidf",
            "accuracy": m_b["accuracy"],
            "f1_macro": m_b["f1_macro"],
            "f1_weighted": m_b["f1_weighted"],
            "best_epoch": None,
            "best_val_macro_f1": None,
            "n_train": len(train_part),
            "n_val": len(val_part),
            "n_test": len(test_df),
        }
    )
    confusion_payload[base_key] = cm_b.tolist()
    report_lines.append(f"## {base_key}\n{m_b['classification_report']}")

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(["f1_macro", "accuracy", "f1_weighted"], ascending=False)
        .reset_index(drop=True)
    )
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    metrics_df.to_csv(out_dir / "leaderboard.csv", index=False)
    (out_dir / "classification_reports.txt").write_text("\n\n".join(report_lines), encoding="utf-8")
    write_json(out_dir / "confusion_matrices.json", {"labels_order": labels_order, "matrices": confusion_payload})

    summary = {
        "task": "Project 3 Task 5 - Deep learning classification",
        "config": {
            "train": args.train,
            "test": args.test,
            "text_col": args.text_col,
            "score_col": args.score_col,
            "label_mode": args.label_mode,
            "features": features,
            "architectures": archs,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_len": args.max_len,
            "val_ratio": args.val_ratio,
            "min_token_freq": args.min_token_freq,
            "max_vocab_tokens": args.max_vocab_tokens,
            "svd_max_features": args.svd_max_features,
            "pmi_max_features": args.pmi_max_features,
            "min_df": args.min_df,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "seed": args.seed,
            "trainable_embeddings": args.trainable_embeddings,
            "device": str(device),
            "smoke": args.smoke,
        },
        "dataset": {
            "n_train_total": int(len(ds.train_df)),
            "n_test_total": int(len(ds.test_df)),
            "n_train_used": int(len(train_part)),
            "n_val_used": int(len(val_part)),
            "n_test_used": int(len(test_df)),
            "labels_order": labels_order,
            "train_label_distribution": train_part["label"].value_counts().to_dict(),
            "val_label_distribution": val_part["label"].value_counts().to_dict(),
            "test_label_distribution": test_df["label"].value_counts().to_dict(),
            "model_vocab_size": len(vocab),
        },
        "feature_metadata": feature_meta,
        "best_overall": metrics_df.iloc[0].to_dict() if not metrics_df.empty else {},
        "baseline": metrics_df[metrics_df["experiment_key"] == base_key].to_dict(orient="records"),
        "artifacts": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "leaderboard_csv": str(out_dir / "leaderboard.csv"),
            "reports_txt": str(out_dir / "classification_reports.txt"),
            "confusion_json": str(out_dir / "confusion_matrices.json"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)

    print("Task 5 completed")
    print(metrics_df[["experiment_key", "accuracy", "f1_macro", "f1_weighted"]].head(16).to_string(index=False))
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
