from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SentimentDataset:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    labels_order: list[str]


def map_labels(scores: pd.Series, mode: str) -> pd.Series:
    s = pd.to_numeric(scores, errors="coerce")
    if mode == "score5":
        return s.astype("Int64").astype(str)
    if mode == "sentiment3":
        out = pd.Series(index=s.index, dtype="object")
        out.loc[s <= 2] = "negative"
        out.loc[s == 3] = "neutral"
        out.loc[s >= 4] = "positive"
        return out
    if mode == "binary":
        out = pd.Series(index=s.index, dtype="object")
        out.loc[s <= 2] = "negative"
        out.loc[s >= 4] = "positive"
        return out
    raise ValueError(f"Unsupported mode: {mode}")


def load_sentiment_dataset(
    train_path: str,
    test_path: str,
    text_col: str,
    score_col: str,
    label_mode: str,
) -> SentimentDataset:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for df in (train_df, test_df):
        if text_col not in df.columns:
            raise ValueError(f"Missing text column '{text_col}' in dataset. Columns: {df.columns.tolist()}")
        if score_col not in df.columns:
            raise ValueError(f"Missing score column '{score_col}' in dataset. Columns: {df.columns.tolist()}")
        df[text_col] = df[text_col].fillna("").astype(str)
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    train_df["label"] = map_labels(train_df[score_col], label_mode)
    test_df["label"] = map_labels(test_df[score_col], label_mode)

    train_df = train_df.dropna(subset=[score_col, "label"]).copy()
    test_df = test_df.dropna(subset=[score_col, "label"]).copy()

    if label_mode == "binary":
        train_df = train_df[train_df[score_col] != 3].copy()
        test_df = test_df[test_df[score_col] != 3].copy()

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    y_train = train_df["label"].astype(str).to_numpy()
    y_test = test_df["label"].astype(str).to_numpy()
    labels_order = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())

    return SentimentDataset(train_df=train_df, test_df=test_df, labels_order=labels_order)
