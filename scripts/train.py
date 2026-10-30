"""Reproducible historical tweet content modeling with separated selection and
test data."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]


def measure(model, features, labels, threshold):
    """Report minority-sensitive ranking and threshold-dependent metrics."""
    scores = model.predict_proba(features)[:, list(model.classes_).index("1")]
    predicted = np.where(scores >= threshold, "1", "0")
    return {
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "weighted_f1": f1_score(labels, predicted, average="weighted"),
        "average_precision": average_precision_score(labels == "1", scores),
        "roc_auc": roc_auc_score(labels == "1", scores),
        "classification_report": classification_report(
            labels, predicted, labels=["0", "1"], output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=["0", "1"]
        ).tolist(),
    }


def load_partitions(data_dir):
    """Group normalized tweets without using row IDs as features."""
    frame = pd.read_csv(data_dir / "train.csv")
    if not frame["id"].is_unique:
        raise ValueError("Source row IDs must be unique.")
    texts = frame["tweet"].fillna("").str.split().str.join(" ")
    if not texts.str.len().between(1, 5000).all():
        raise ValueError("Unexpected empty or oversized source text")
    normalized = texts.map(lambda text: re.sub(r"\W+", " ", text.lower()).strip())
    groups = pd.factorize(normalized)[0]
    develop, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            texts, groups=groups
        )
    )
    train_rel, valid_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(
            texts.iloc[develop], groups=groups[develop]
        )
    )
    training, valid = develop[train_rel], develop[valid_rel]
    for left, right in [(training, valid), (training, test), (valid, test)]:
        if not set(groups[left]).isdisjoint(groups[right]):
            raise ValueError("Duplicate text overlap between partitions")
    return (
        texts,
        frame["label"].astype(str),
        {"train": training, "validation": valid, "test": test},
        "grouped normalized text 60/20/20; not temporal or author-separated",
        {},
        len(frame),
    )


def train_models(data_dir, output_dir):
    """Select by validation AP and choose threshold on validation class-1 F1."""
    features, labels, partitions, strategy, detail, original_rows = load_partitions(
        data_dir
    )
    training, validation, test = (
        partitions[key] for key in ["train", "validation", "test"]
    )
    candidates = {
        name: make_pipeline(
            TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
            ),
            classifier,
        )
        for name, classifier in [
            ("majority_baseline", DummyClassifier(strategy="most_frequent")),
            ("multinomial_naive_bayes", MultinomialNB(alpha=0.5)),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    }
    comparison = {}
    for name, model in candidates.items():
        print("Fitting", name, flush=True)
        model.fit(features.iloc[training], labels.iloc[training])
        comparison[name] = measure(
            model, features.iloc[validation], labels.iloc[validation], 0.5
        )
    winner = max(comparison, key=lambda key: comparison[key]["average_precision"])
    model = candidates[winner]
    valid_scores = model.predict_proba(features.iloc[validation])[
        :, list(model.classes_).index("1")
    ]
    thresholds = np.arange(0.01, 1, 0.01)
    threshold = float(
        thresholds[
            np.argmax(
                [
                    f1_score(labels.iloc[validation] == "1", valid_scores >= value)
                    for value in thresholds
                ]
            )
        ]
    )
    print("Selected", winner, "threshold", threshold, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pipeline.joblib"
    joblib.dump(model, path, compress=3)
    report = {
        "version": "1.0.0",
        "selected_model": winner,
        "threshold": threshold,
        "input_field": "tweet",
        "classes": ["0", "1"],
        "original_rows": original_rows,
        "source_sha256": hashlib.sha256(
            (data_dir / "train.csv").read_bytes()
        ).hexdigest(),
        "label_source": (
            "Supplied historical racist/sexist binary labels; "
            "provenance not independently verified"
        ),
        "split_strategy": strategy,
        "split_detail": detail,
        "split": {
            name: {
                "rows": len(idx),
                "class_counts": labels.iloc[idx].value_counts().to_dict(),
            }
            for name, idx in partitions.items()
        },
        "validation_candidates": comparison,
        "validation_selected_threshold": measure(
            model, features.iloc[validation], labels.iloc[validation], threshold
        ),
        "test": measure(model, features.iloc[test], labels.iloc[test], threshold),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    sample = {
        "tweet": "I enjoy learning Python and building useful projects with friends."
    }
    (output_dir / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    print(json.dumps({"winner": winner, "test": report["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    train_models(args.data_dir, args.output_dir)
