"""Text, threshold, HTTP and integrity checks for the manual-release example."""

import io
from pathlib import Path
import pytest
from tweet_classifier.service import PredictionService, prepare_records
from tweet_classifier.web import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="service", scope="module")
def fixture_service():
    """Load one reviewed pipeline."""
    return PredictionService(ROOT / "models")


@pytest.fixture(name="client")
def fixture_client():
    """Create an isolated test client."""
    return create_app({"TESTING": True, "MODEL_DIR": ROOT / "models"}).test_client()


@pytest.mark.parametrize(
    "records",
    [
        None,
        {},
        [],
        [None],
        [{}],
        [{"tweet": ""}],
        [{"tweet": 42}],
        [{"tweet": "x" * 5001}],
        [{"tweet": "x"}] * 201,
    ],
)
def test_invalid_records(records):
    """Reject missing, malformed and excessive text inputs."""
    with pytest.raises(ValueError):
        prepare_records(records)


@pytest.mark.parametrize(
    "threshold", [True, -1, 0, 1, 2, "bad", float("nan"), float("inf")]
)
def test_bad_threshold(service, threshold):
    """Thresholds must be bounded numeric values."""
    with pytest.raises(ValueError):
        service.predict([service.sample], threshold)


def test_inference_and_threshold(service):
    """Batch scoring agrees and lower thresholds never reduce flags."""
    item = service.predict([service.sample])[0]
    assert service.predict([service.sample] * 3) == [item] * 3
    assert abs(sum(item["scores"].values()) - 1) < 0.00001
    low = service.predict([service.sample], 0.01)[0]
    high = service.predict([service.sample], 0.99)[0]
    assert int(low["category"]) >= int(high["category"])
    assert low["scores"] == high["scores"]


def test_unseen_words(service):
    """Unknown vocabulary is disclosed instead of implying understood meaning."""
    assert service.predict([{"tweet": "zzzzqxwwzz"}])[0]["warning"]


def test_app_api_and_csv(client, service):
    """All entry points use the same pipeline and identify the real task."""
    home = client.get("/")
    assert home.status_code == 200 and b"Tweet Content Lab" in home.data
    assert b"not positive/negative sentiment" in home.data
    result = client.post("/api/predict", json={"records": [service.sample]})
    assert result.json["predictions"] == service.predict([service.sample])
    assert "not generic sentiment" in result.json["evaluation_target"]
    html = client.post("/predict", data=dict(service.sample, threshold=0.2))
    assert html.status_code == 200 and b"LEARNED MODEL" in html.data
    sample = client.get("/sample.csv")
    result = client.post(
        "/predict-csv", data={"file": (io.BytesIO(sample.data), "sample.csv")}
    )
    assert result.status_code == 200
    assert service.sample["tweet"].encode() not in result.data


def test_errors_limits_and_escaping(client):
    """Malformed requests, excessive bodies and HTML injection fail safely."""
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/predict-csv").status_code == 400
    assert client.get("/missing").status_code == 404
    assert (
        client.post(
            "/api/predict",
            data=b"x" * (1024 * 1024 + 1),
            content_type="application/json",
        ).status_code
        == 413
    )
    assert (
        b"<script>alert(1)</script>"
        not in client.post("/predict", data={"tweet": "<script>alert(1)</script>"}).data
    )


def test_report_and_headers(client):
    """Metric totals and readiness describe the reviewed source-label model."""
    report = client.get("/api/metrics").json
    assert sum(v["rows"] for v in report["split"].values()) == 31962
    assert sum(map(sum, report["test"]["confusion_matrix"])) == report["test"]["rows"]
    response = client.get("/health/ready")
    assert response.json["status"] == "ready"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_integrity(tmp_path):
    """Reject unexpected serialized artifacts before deserialization."""
    (tmp_path / "metrics.json").write_bytes((ROOT / "models/metrics.json").read_bytes())
    (tmp_path / "pipeline.joblib").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService(tmp_path)
