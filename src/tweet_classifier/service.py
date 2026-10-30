"""Bounded tweet input and threshold-based classification with trusted artifacts."""

import hashlib
import json
from pathlib import Path

import joblib


def prepare_records(records: object) -> list[str]:
    """Normalize whitespace, retaining text rather than deleting negation."""
    if not isinstance(records, list) or not 1 <= len(records) <= 200:
        raise ValueError("Provide 1 to 200 tweet records.")
    texts = []
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get("tweet"), str):
            raise ValueError(f"Row {number}: tweet must be a string.")
        text = " ".join(record["tweet"].split())
        if not text or len(text) > 5000:
            raise ValueError(f"Row {number}: provide 1 to 5000 characters.")
        texts.append(text)
    return texts


class PredictionService:
    """Load one reviewed text model and disclose threshold/class-score limits."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self.report = json.loads((model_dir / "metrics.json").read_text())
        path = model_dir / "pipeline.joblib"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != self.report["artifact_sha256"]
        ):
            raise ValueError("Model artifact integrity check failed.")
        self.model = joblib.load(path)
        self.sample = json.loads((model_dir / "sample.json").read_text())

    def predict(
        self, records: object, threshold: float | str | None = None
    ) -> list[dict]:
        """Return source labels, uncalibrated scores and vocabulary warnings."""
        if threshold is None:
            threshold = self.report["threshold"]
        try:
            if isinstance(threshold, bool):
                raise ValueError
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("Threshold must be a number from 0.01 to 0.99.") from exc
        if not 0.01 <= threshold <= 0.99:
            raise ValueError("Threshold must be from 0.01 to 0.99.")
        texts = prepare_records(records)
        scores = self.model.predict_proba(texts)
        vectorizer = self.model.named_steps["tfidfvectorizer"]
        results = []
        for text, row in zip(texts, scores):
            positive = float(row[list(self.model.classes_).index("1")])
            label = "1" if positive >= threshold else "0"
            terms = vectorizer.build_analyzer()(text)
            matched = sum(term in vectorizer.vocabulary_ for term in terms)
            results.append(
                {
                    "category": label,
                    "threshold": threshold,
                    "description": (
                        "Source-labeled racist/sexist content"
                        if label == "1"
                        else "Other source label"
                    ),
                    "scores": {
                        str(key): round(float(value), 6)
                        for key, value in zip(self.model.classes_, row)
                    },
                    "matched_terms": matched,
                    "warning": (
                        "No vocabulary overlap; unreliable result."
                        if not matched
                        else None
                    ),
                }
            )
        return results
