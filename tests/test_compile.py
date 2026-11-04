"""The pipeline must compile to valid KFP IR locally — compilation is the
contract Vertex AI consumes, and it needs no GCP account."""

import yaml

from pipeline.pipeline import compile_pipeline


def test_pipeline_compiles_with_all_components_and_gate(tmp_path):
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(str(out))

    text = out.read_text()
    doc = yaml.safe_load(text)

    assert doc["pipelineInfo"]["name"] == "customer-churn-training"
    assert doc["schemaVersion"]  # IR document, not an empty file

    # All five steps are present in the compiled graph.
    for component in ["ingest", "train", "evaluate", "metric-gate", "register-model"]:
        assert component in text, f"missing component in IR: {component}"

    # The conditional deploy: register-model sits behind a trigger policy
    # (the compiled form of dsl.If) named after the accuracy gate.
    assert "triggerPolicy" in text
    assert "accuracy-gate-passed" in text
