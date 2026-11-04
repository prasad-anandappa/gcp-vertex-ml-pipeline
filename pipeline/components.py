"""Pipeline components.

Each component is a self-contained lightweight Python component (all imports
inside the function body): the KFP compiler serialises the function source
into the pipeline IR, so what compiles here is exactly what Vertex AI would
execute in its own containers — no hidden local modules.

The component logic is unit-tested locally by calling the underlying
functions directly with file-backed artifact stubs (see tests/).
"""

from kfp import dsl

# Feature/target columns are repeated inside each component body on purpose:
# a serialised component cannot rely on module-level names existing in the
# container that eventually executes it.
FEATURE_COLUMNS = ("age", "income", "visits_per_month", "tenure_months")
TARGET_COLUMN = "churned"


@dsl.component(base_image="python:3.12", packages_to_install=["pandas>=2.2"])
def ingest(csv_uri: str, dataset: dsl.Output[dsl.Dataset]) -> int:
    """Reads the source CSV, validates the expected schema, drops incomplete
    rows, and writes the cleaned dataset artifact.

    ``csv_uri`` may be a local path (local runs/tests) or a ``gs://`` URI
    (Vertex runs, where pandas reads it via gcsfs).
    """
    import pandas as pd

    df = pd.read_csv(csv_uri)
    required = {"age", "income", "visits_per_month", "tenure_months", "churned"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    df.to_csv(dataset.path, index=False)

    dataset.metadata["rows"] = len(df)
    dataset.metadata["dropped_rows"] = before - len(df)
    dataset.metadata["source"] = csv_uri
    return len(df)


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas>=2.2", "scikit-learn>=1.5"],
)
def train(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    test_size: float = 0.25,
    random_state: int = 42,
) -> float:
    """Trains a RandomForest churn classifier and serialises it as the model
    artifact. Returns training accuracy (evaluation on held-out data is the
    evaluate component's job — training accuracy is reported, never gated on).
    """
    import pickle

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    features = ["age", "income", "visits_per_month", "tenure_months"]
    df = pd.read_csv(dataset.path)
    x, y = df[features], df["churned"]
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    clf.fit(x_train, y_train)

    with open(model.path, "wb") as fh:
        pickle.dump(clf, fh)

    model.metadata["framework"] = "scikit-learn RandomForestClassifier"
    model.metadata["features"] = ",".join(features)
    model.metadata["train_rows"] = len(x_train)
    return float(clf.score(x_train, y_train))


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas>=2.2", "scikit-learn>=1.5"],
)
def evaluate(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Input[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
    test_size: float = 0.25,
    random_state: int = 42,
) -> float:
    """Scores the model on the held-out split (same seed/split as train, so
    the rows evaluated here are exactly the rows training never saw) and logs
    accuracy + ROC AUC to the metrics artifact. Returns accuracy — the value
    the metric gate consumes.
    """
    import pickle

    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    features = ["age", "income", "visits_per_month", "tenure_months"]
    df = pd.read_csv(dataset.path)
    x, y = df[features], df["churned"]
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with open(model.path, "rb") as fh:
        clf = pickle.load(fh)

    predictions = clf.predict(x_test)
    probabilities = clf.predict_proba(x_test)[:, 1]
    accuracy = float(accuracy_score(y_test, predictions))
    auc = float(roc_auc_score(y_test, probabilities))

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("roc_auc", auc)
    metrics.log_metric("test_rows", len(x_test))
    return accuracy


@dsl.component(base_image="python:3.12")
def metric_gate(accuracy: float, threshold: float) -> bool:
    """The deploy gate: True only when held-out accuracy meets the threshold.

    A separate component (rather than an inline condition) so the decision is
    a named, logged, unit-testable step in the pipeline graph.
    """
    passed = accuracy >= threshold
    print(
        f"metric gate: accuracy={accuracy:.4f} threshold={threshold:.4f} "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return passed


@dsl.component(base_image="python:3.12")
def register_model(
    model: dsl.Input[dsl.Model],
    accuracy: float,
    manifest: dsl.Output[dsl.Artifact],
) -> str:
    """Conditional deploy step — runs only when the metric gate passes.

    Writes a deployment manifest describing the model that cleared the gate.
    In a credentialed Vertex AI environment this step additionally calls
    ``google.cloud.aiplatform.Model.upload`` + ``Endpoint.deploy`` (see the
    README's "Going to Vertex AI" section); here it deliberately stops at the
    manifest so the pipeline stays runnable and testable with no GCP account,
    and so a model can never be "deployed" by accident from a laptop.
    """
    import json
    from datetime import datetime, timezone

    doc = {
        "model_uri": model.uri,
        "accuracy": accuracy,
        "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": "Vertex AI Model Registry",
        "status": "manifest-only: no GCP credentials in this environment",
    }
    with open(manifest.path, "w") as fh:
        json.dump(doc, fh, indent=2)

    manifest.metadata["accuracy"] = accuracy
    manifest.metadata["target"] = doc["target"]
    return doc["status"]
