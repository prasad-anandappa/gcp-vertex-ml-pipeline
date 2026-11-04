"""Unit tests for the component logic.

The KFP components are self-contained functions; these tests call the
underlying Python functions directly with file-backed artifact stubs, so the
exact code Vertex AI would execute is what gets verified.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from pipeline.components import (
    evaluate,
    ingest,
    metric_gate,
    register_model,
    train,
)

DATA = Path(__file__).parent.parent / "data" / "customers.csv"


class ArtifactStub:
    """Stands in for a KFP artifact: just path / uri / metadata."""

    def __init__(self, path):
        self.path = str(path)
        self.uri = f"file://{path}"
        self.metadata = {}


class MetricsStub(ArtifactStub):
    def log_metric(self, name, value):
        self.metadata[name] = value


@pytest.fixture()
def stubs(tmp_path):
    return {
        "dataset": ArtifactStub(tmp_path / "clean.csv"),
        "model": ArtifactStub(tmp_path / "model.pkl"),
        "metrics": MetricsStub(tmp_path / "metrics.json"),
        "manifest": ArtifactStub(tmp_path / "manifest.json"),
    }


def test_ingest_cleans_and_counts(stubs):
    rows = ingest.python_func(csv_uri=str(DATA), dataset=stubs["dataset"])
    assert rows == 1200
    assert stubs["dataset"].metadata["rows"] == 1200
    written = pd.read_csv(stubs["dataset"].path)
    assert len(written) == 1200
    assert list(written.columns) == [
        "age",
        "income",
        "visits_per_month",
        "tenure_months",
        "churned",
    ]


def test_ingest_rejects_missing_columns(stubs, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("age,income\n30,50000\n")
    with pytest.raises(ValueError, match="missing required columns"):
        ingest.python_func(csv_uri=str(bad), dataset=stubs["dataset"])


def test_train_produces_loadable_model(stubs):
    ingest.python_func(csv_uri=str(DATA), dataset=stubs["dataset"])
    train_accuracy = train.python_func(
        dataset=stubs["dataset"], model=stubs["model"],
        test_size=0.25, random_state=42,
    )
    assert 0.5 < train_accuracy <= 1.0
    with open(stubs["model"].path, "rb") as fh:
        model = pickle.load(fh)
    assert hasattr(model, "predict")
    assert stubs["model"].metadata["train_rows"] == 900


def test_evaluate_beats_majority_baseline(stubs):
    ingest.python_func(csv_uri=str(DATA), dataset=stubs["dataset"])
    train.python_func(
        dataset=stubs["dataset"], model=stubs["model"],
        test_size=0.25, random_state=42,
    )
    accuracy = evaluate.python_func(
        dataset=stubs["dataset"], model=stubs["model"],
        metrics=stubs["metrics"], test_size=0.25, random_state=42,
    )
    df = pd.read_csv(DATA)
    majority_baseline = 1.0 - df["churned"].mean()
    assert accuracy > majority_baseline
    assert accuracy >= 0.75  # the bundled data is designed to clear the gate
    assert stubs["metrics"].metadata["test_rows"] == 300
    assert 0.0 <= stubs["metrics"].metadata["roc_auc"] <= 1.0


@pytest.mark.parametrize(
    "accuracy,threshold,expected",
    [(0.90, 0.75, True), (0.75, 0.75, True), (0.7499, 0.75, False), (0.0, 0.75, False)],
)
def test_metric_gate(accuracy, threshold, expected):
    assert metric_gate.python_func(accuracy=accuracy, threshold=threshold) is expected


def test_register_model_writes_manifest(stubs):
    status = register_model.python_func(
        model=stubs["model"], accuracy=0.8167, manifest=stubs["manifest"]
    )
    doc = json.loads(Path(stubs["manifest"].path).read_text())
    assert doc["accuracy"] == 0.8167
    assert doc["target"] == "Vertex AI Model Registry"
    assert "manifest-only" in doc["status"]
    assert status == doc["status"]


def test_full_chain_gate_decides(stubs):
    """End to end through the component functions: ingest -> train ->
    evaluate -> gate. The same accuracy passes a 0.75 gate and fails a
    0.90 gate — the conditional deploy step keys off exactly this."""
    ingest.python_func(csv_uri=str(DATA), dataset=stubs["dataset"])
    train.python_func(
        dataset=stubs["dataset"], model=stubs["model"],
        test_size=0.25, random_state=42,
    )
    accuracy = evaluate.python_func(
        dataset=stubs["dataset"], model=stubs["model"],
        metrics=stubs["metrics"], test_size=0.25, random_state=42,
    )
    assert metric_gate.python_func(accuracy=accuracy, threshold=0.75) is True
    assert metric_gate.python_func(accuracy=accuracy, threshold=0.90) is False
